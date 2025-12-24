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

template <typename Float, typename Spectrum>
class DistantAngularSensor final : public Sensor<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Sensor, m_film, sample_wavelengths)
    MI_IMPORT_TYPES(Scene, Shape)

    DistantAngularSensor(const Properties &props) : Base(props) {
        // Get target point (default to origin)
        m_target_point = props.get<ScalarPoint3f>("target", ScalarPoint3f(0.f, 0.f, 0.f));
        
        // Store film resolution for angular mapping
        m_film_size = m_film->size();
        
        // Check reconstruction filter radius
        if (m_film->rfilter()->radius() > 0.5f + math::RayEpsilon<Float>) {
            Log(Warn, "This sensor should be used with a reconstruction filter "
                      "with a radius of 0.5 or lower (e.g. box filter).");
        }
    }

    void set_scene(const Scene *scene) override {
        m_bsphere = scene->bbox().bounding_sphere();
        m_bsphere.radius = dr::maximum(math::RayEpsilon<Float>,
                                       m_bsphere.radius * (1.f + math::RayEpsilon<Float>));
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &film_sample,
                                          const Point2f & /*aperture_sample*/,
                                          Mask active) const override {
        MI_MASK_ARGUMENT(active);

        Ray3f ray;
        ray.time = time;

        // Sample spectrum
        auto [wavelengths, wav_weight] = sample_wavelengths(
            dr::zeros<SurfaceInteraction3f>(), wavelength_sample, active);
        ray.wavelengths = wavelengths;

        // Snap to bin center: all rays within a pixel use the SAME direction
        // film_sample is in [0, 1], pixel index = floor(film_sample * size)
        // bin center = (pixel_index + 0.5) / size
        Float pixel_x = dr::floor(film_sample.x() * Float(m_film_size.x()));
        Float pixel_y = dr::floor(film_sample.y() * Float(m_film_size.y()));
        
        // Clamp to valid range (edge case when film_sample == 1.0)
        pixel_x = dr::minimum(pixel_x, Float(m_film_size.x() - 1));
        pixel_y = dr::minimum(pixel_y, Float(m_film_size.y() - 1));
        
        // Compute bin center in [0, 1]
        Float center_x = (pixel_x + 0.5f) / Float(m_film_size.x());
        Float center_y = (pixel_y + 0.5f) / Float(m_film_size.y());
        
        // Map to angles: x -> phi [0, 2*pi], y -> theta [0, pi]
        Float phi = center_x * (2.f * dr::Pi<Float>);
        Float theta = center_y * dr::Pi<Float>;
        
        // Compute outgoing direction omega = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
        auto [sin_theta, cos_theta] = dr::sincos(theta);
        auto [sin_phi, cos_phi] = dr::sincos(phi);
        
        Vector3f omega(sin_theta * cos_phi, 
                       sin_theta * sin_phi, 
                       cos_theta);
        
        // Ray direction is opposite to outgoing direction (ray travels into scene)
        ray.d = -omega;
        
        // Ray origin: start from behind the target, looking toward it
        // ray.o = target + omega * 2 * radius
        ray.o = m_target_point + omega * (2.f * m_bsphere.radius);

        return { ray, wav_weight & active };
    }

    std::pair<RayDifferential3f, Spectrum> sample_ray_differential(
        Float time, Float wavelength_sample, const Point2f &film_sample,
        const Point2f &aperture_sample, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        RayDifferential3f ray;
        Spectrum ray_weight;

        std::tie(ray, ray_weight) = sample_ray(
            time, wavelength_sample, film_sample, aperture_sample, active);

        // No differentials since all rays in a pixel have the same direction
        ray.has_differentials = false;

        return { ray, ray_weight & active };
    }

    // This sensor does not occupy any particular region of space
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "DistantAngularSensor[" << std::endl
            << "  film = " << m_film << "," << std::endl
            << "  film_size = " << m_film_size << "," << std::endl
            << "  target = " << m_target_point << "," << std::endl
            << "  bsphere = " << m_bsphere << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()

protected:
    ScalarBoundingSphere3f m_bsphere;
    ScalarPoint3f m_target_point;
    ScalarPoint2i m_film_size;
};

MI_EXPORT_PLUGIN(DistantAngularSensor)

NAMESPACE_END(mitsuba)