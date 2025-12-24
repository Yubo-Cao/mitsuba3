/**
 * Batched Angular Distant Sensor for Mitsuba 3
 *
 * This sensor maps film pixels to spherical angular directions, allowing
 * measurement of far-field radiance across multiple directions in a single
 * render call. This avoids kernel recompilation that occurs when using
 * multiple individual distant sensors.
 *
 * Film coordinates (x, y) map to:
 *   - x -> phi (azimuthal angle): [0, 2π)
 *   - y -> theta (polar angle): [0, π]
 *
 * The direction is computed as: ω = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))
 * Rays are shot in the -ω direction to measure radiance leaving the scene
 * in direction +ω.
 */

#include <mitsuba/core/bbox.h>
#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/shape.h>

NAMESPACE_BEGIN(mitsuba)

enum class AngularTargetType { Shape, Point, None };

template <typename Float, typename Spectrum, AngularTargetType TargetType>
class DistantAngularImpl;

/**!
.. _sensor-distant_angular:

Batched Angular Distant Sensor (:monosp:`distant_angular`)
----------------------------------------------------------

.. pluginparameters::

 * - target
   - |point| or nested :paramtype:`shape` plugin
   - *Optional.* Define the ray target. If unset, rays target the scene's
     bounding sphere center. If a |point| is passed, rays will target it.
     If a shape plugin is passed, ray target points will be sampled from
     its surface.

 * - phi_bins
   - |int|
   - Number of azimuthal angle bins (film width). Default: 16

 * - theta_bins
   - |int|
   - Number of polar angle bins (film height). Default: 8

 * - theta_min
   - |float|
   - Minimum polar angle in radians. Default: 0

 * - theta_max
   - |float|
   - Maximum polar angle in radians. Default: π

 * - phi_min
   - |float|
   - Minimum azimuthal angle in radians. Default: 0

 * - phi_max
   - |float|
   - Maximum azimuthal angle in radians. Default: 2π

This sensor measures far-field radiance across a grid of angular directions
in a single render call. Each pixel in the output film corresponds to a
specific (phi, theta) direction.

The sensor shoots rays from outside the scene toward the target, with ray
directions covering a hemisphere or full sphere depending on theta range.
*/

template <typename Float, typename Spectrum>
class DistantAngular final : public Sensor<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Sensor, m_film)
    MI_IMPORT_TYPES(Scene, Shape)

    DistantAngular(const Properties &props) : Base(props), m_props(props) {
        // Determine target type
        if (props.has_property("target")) {
            if (props.type("target") == Properties::Type::Vector) {
                props.get<ScalarPoint3f>("target");
                m_target_type = AngularTargetType::Point;
            } else if (props.type("target") == Properties::Type::Object) {
                m_target_type = AngularTargetType::Shape;
            } else {
                Throw("Unsupported 'target' parameter type");
            }
        } else {
            m_target_type = AngularTargetType::None;
        }

        props.mark_queried("target");
        props.mark_queried("phi_bins");
        props.mark_queried("theta_bins");
        props.mark_queried("theta_min");
        props.mark_queried("theta_max");
        props.mark_queried("phi_min");
        props.mark_queried("phi_max");
    }

    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    template <AngularTargetType TType>
    using Impl = DistantAngularImpl<Float, Spectrum, TType>;

    std::vector<ref<Object>> expand() const override {
        ref<Object> result;
        switch (m_target_type) {
            case AngularTargetType::Shape:
                result = (Object *) new Impl<AngularTargetType::Shape>(m_props);
                break;
            case AngularTargetType::Point:
                result = (Object *) new Impl<AngularTargetType::Point>(m_props);
                break;
            case AngularTargetType::None:
                result = (Object *) new Impl<AngularTargetType::None>(m_props);
                break;
            default:
                Throw("Unsupported target type!");
        }
        return { result };
    }

    MI_DECLARE_CLASS()

protected:
    Properties m_props;
    AngularTargetType m_target_type;
};

template <typename Float, typename Spectrum, AngularTargetType TargetType>
class DistantAngularImpl final : public Sensor<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Sensor, m_film, sample_wavelengths)
    MI_IMPORT_TYPES(Scene, Shape)

    DistantAngularImpl(const Properties &props) : Base(props) {
        // Angular range parameters
        m_phi_min = props.get<ScalarFloat>("phi_min", 0.f);
        m_phi_max =
            props.get<ScalarFloat>("phi_max", 2.f * dr::Pi<ScalarFloat>);
        m_theta_min = props.get<ScalarFloat>("theta_min", 0.f);
        m_theta_max = props.get<ScalarFloat>("theta_max", dr::Pi<ScalarFloat>);

        // Film dimensions define angular resolution
        ScalarPoint2i film_size = m_film->size();
        m_phi_bins              = film_size.x();
        m_theta_bins            = film_size.y();

        Log(Info, "DistantAngular: %d phi bins x %d theta bins", m_phi_bins,
            m_theta_bins);
        Log(Info, "  phi: [%.3f, %.3f], theta: [%.3f, %.3f]", m_phi_min,
            m_phi_max, m_theta_min, m_theta_max);

        // Check reconstruction filter
        if (m_film->rfilter()->radius() > 0.5f + math::RayEpsilon<Float>) {
            Log(Warn,
                "This sensor should be used with a box filter (radius 0.5)");
        }

        // Set up target
        if constexpr (TargetType == AngularTargetType::Point) {
            m_target_point = props.get<ScalarPoint3f>("target");
        } else if constexpr (TargetType == AngularTargetType::Shape) {
            auto obj       = props.get<ref<Object>>("target");
            m_target_shape = dynamic_cast<Shape *>(obj.get());
            if (!m_target_shape)
                Throw("Invalid target parameter, must be Point3f or Shape");
        }
    }

    void set_scene(const Scene *scene) override {
        m_bsphere = scene->bbox().bounding_sphere();
        m_bsphere.radius =
            dr::maximum(math::RayEpsilon<Float>,
                        m_bsphere.radius * (1.f + math::RayEpsilon<Float>) );
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &film_sample,
                                          const Point2f &aperture_sample,
                                          Mask active) const override {
        MI_MASK_ARGUMENT(active);

        Ray3f ray;
        ray.time = time;

        // Sample wavelengths
        auto [wavelengths, wav_weight] = sample_wavelengths(
            dr::zeros<SurfaceInteraction3f>(), wavelength_sample, active);
        ray.wavelengths = wavelengths;

        // Convert film sample to angular direction
        Float phi = m_phi_min + film_sample.x() * (m_phi_max - m_phi_min);
        Float theta =
            m_theta_min + film_sample.y() * (m_theta_max - m_theta_min);

        Float sin_theta = dr::sin(theta);
        Float cos_theta = dr::cos(theta);
        Float sin_phi   = dr::sin(phi);
        Float cos_phi   = dr::cos(phi);

        Vector3f omega(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        ray.d = -omega;

        // Sample target and position ray origin
        Spectrum ray_weight = 0.f;

        if constexpr (TargetType == AngularTargetType::Point) {
            ray.o      = m_target_point - 2.f * ray.d * m_bsphere.radius;
            ray_weight = wav_weight;
        } else if constexpr (TargetType == AngularTargetType::Shape) {
            PositionSample3f ps =
                m_target_shape->sample_position(time, aperture_sample, active);
            ray.o      = ps.p - 2.f * ray.d * m_bsphere.radius;
            ray_weight = wav_weight / (ps.pdf * m_target_shape->surface_area());
        } else {
            // Target bounding sphere center with offset for aperture
            Point2f offset =
                warp::square_to_uniform_disk_concentric(aperture_sample);

            // Create local coordinate system around ray direction
            auto abs_z     = dr::abs(ray.d.z());
            Vector3f up    = dr::select(abs_z < 0.99f, Vector3f(0.f, 0.f, 1.f),
                                        Vector3f(1.f, 0.f, 0.f));
            Vector3f right = dr::normalize(dr::cross(up, ray.d));
            up             = dr::cross(ray.d, right);

            Vector3f perp_offset = right * offset.x() + up * offset.y();
            ray.o = m_bsphere.center + perp_offset * m_bsphere.radius -
                    ray.d * m_bsphere.radius;
            ray_weight = wav_weight;
        }

        return { ray, ray_weight & active };
    }

    std::pair<RayDifferential3f, Spectrum> sample_ray_differential(
        Float time, Float wavelength_sample, const Point2f &film_sample,
        const Point2f &aperture_sample, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        RayDifferential3f ray;
        Spectrum ray_weight;

        std::tie(ray, ray_weight) = sample_ray(
            time, wavelength_sample, film_sample, aperture_sample, active);

        ray.has_differentials = false;

        return { ray, ray_weight & active };
    }

    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "DistantMeow[" << std::endl
            << "  phi_bins = " << m_phi_bins << "," << std::endl
            << "  theta_bins = " << m_theta_bins << "," << std::endl
            << "  phi = [" << m_phi_min << ", " << m_phi_max << "],"
            << std::endl
            << "  theta = [" << m_theta_min << ", " << m_theta_max << "],"
            << std::endl
            << "  film = " << m_film << "," << std::endl;

        if constexpr (TargetType == AngularTargetType::Point)
            oss << "  target = " << m_target_point << std::endl;
        else if constexpr (TargetType == AngularTargetType::Shape)
            oss << "  target = " << m_target_shape << std::endl;
        else
            oss << "  target = bsphere_center" << std::endl;

        oss << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()

protected:
    ScalarBoundingSphere3f m_bsphere;
    ref<Shape> m_target_shape;
    Point3f m_target_point;

    ScalarFloat m_phi_min, m_phi_max;
    ScalarFloat m_theta_min, m_theta_max;
    int m_phi_bins, m_theta_bins;
};

MI_EXPORT_PLUGIN(DistantAngular)

NAMESPACE_END(mitsuba)