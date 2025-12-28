/**
 * Average-Based Angular Distant Sensor for Mitsuba 3
 *
 * This sensor maps film pixels to spherical angular directions, measuring
 * far-field radiance AVERAGED over each angular bin. Rays within a pixel
 * are distributed across the bin's solid angle extent.
 *
 * Film coordinates map to:
 *   - x (columns) -> phi (azimuthal angle): [0, 2π)
 *   - y (rows) -> theta (polar angle): arccos-spaced from 0 to π
 *
 * Theta bins use arccos splitting: cos(theta) linearly spaced from 1 to -1,
 * then theta = arccos(cos_theta). This gives uniform solid angle bins.
 *
 * Output image shape: [theta_bins, phi_bins] (y=theta, x=phi)
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

template <typename Float, typename Spectrum>
class DistantMeow final : public Sensor<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Sensor, m_film, sample_wavelengths)
    MI_IMPORT_TYPES(Scene, Shape)

    DistantMeow(const Properties &props) : Base(props) {
        // Get target point (default to origin)
        m_target_point = props.get<ScalarPoint3f>("target", ScalarPoint3f(0.f, 0.f, 0.f));

        // Film dimensions: y=theta (rows), x=phi (columns)
        m_film_size = m_film->size();
        m_phi_bins = m_film_size.x();
        m_theta_bins = m_film_size.y();

        Log(Info, "DistantMeow (average-based): %d theta bins x %d phi bins",
            m_theta_bins, m_phi_bins);

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

        // Use film_sample directly (no snapping to bin center)
        // This averages over the bin's angular extent
        Float sample_x = film_sample.x();
        Float sample_y = film_sample.y();

        // Map x -> phi: [0, 2π)
        Float phi = sample_x * (2.f * dr::Pi<Float>);

        // Map y -> theta using arccos splitting:
        // cos_theta linearly from 1 (y=0) to -1 (y=1), then theta = arccos(cos_theta)
        Float cos_theta = 1.f - 2.f * sample_y;
        Float sin_theta = dr::sqrt(dr::maximum(0.f, 1.f - cos_theta * cos_theta));

        auto [sin_phi, cos_phi] = dr::sincos(phi);

        // Compute outgoing direction omega = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
        Vector3f omega(sin_theta * cos_phi,
                       sin_theta * sin_phi,
                       cos_theta);

        // Ray direction is opposite to outgoing direction (ray travels into scene)
        ray.d = -omega;

        // Ray origin: start from behind the target, looking toward it
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

        // No differentials for distant sensor
        ray.has_differentials = false;

        return { ray, ray_weight & active };
    }

    // This sensor does not occupy any particular region of space
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "DistantMeow[" << std::endl
            << "  sampling = average," << std::endl
            << "  theta_bins = " << m_theta_bins << "," << std::endl
            << "  phi_bins = " << m_phi_bins << "," << std::endl
            << "  film = " << m_film << "," << std::endl
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
    int m_phi_bins, m_theta_bins;
};

MI_EXPORT_PLUGIN(DistantMeow)

NAMESPACE_END(mitsuba)
