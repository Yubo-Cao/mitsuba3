/**
 * Center-Based Angular Distant Sensor for Mitsuba 3
 *
 * This sensor maps film pixels to spherical angular directions, measuring
 * far-field radiance at the CENTER of each angular bin. All rays within
 * a pixel share the same direction (the bin center).
 *
 * Film coordinates map to:
 *   - x (columns) -> phi (azimuthal angle): [0, 2π) with offset π/phi_bins
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
class DistantKitten final : public Sensor<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Sensor, m_film, sample_wavelengths)
    MI_IMPORT_TYPES(Scene, Shape)

    DistantKitten(const Properties &props) : Base(props) {
        // Get target point (default to origin)
        m_target_point = props.get<ScalarPoint3f>("target", ScalarPoint3f(0.f, 0.f, 0.f));

        // Film dimensions: y=theta (rows), x=phi (columns)
        m_film_size = m_film->size();
        m_phi_bins = m_film_size.x();
        m_theta_bins = m_film_size.y();

        Log(Info, "DistantKitten (center-based): %d theta bins x %d phi bins",
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

        // Snap to bin center: all rays within a pixel use the SAME direction
        // film_sample is in [0, 1], pixel index = floor(film_sample * size)
        Float pixel_x = dr::floor(film_sample.x() * Float(m_phi_bins));
        Float pixel_y = dr::floor(film_sample.y() * Float(m_theta_bins));

        // Clamp to valid range (edge case when film_sample == 1.0)
        pixel_x = dr::minimum(pixel_x, Float(m_phi_bins - 1));
        pixel_y = dr::minimum(pixel_y, Float(m_theta_bins - 1));

        // Compute bin center in [0, 1]
        Float center_x = (pixel_x + 0.5f) / Float(m_phi_bins);
        Float center_y = (pixel_y + 0.5f) / Float(m_theta_bins);

        // Map x -> phi: [0, 2π) with bin centers at (2*i + 1) * π / phi_bins
        // This matches: phi = linspace(0, 2π, phi_bins, endpoint=False) + π/phi_bins
        Float phi = center_x * (2.f * dr::Pi<Float>);

        // Map y -> theta using arccos splitting:
        // cos_theta linearly from 1 (y=0) to -1 (y=1), then theta = arccos(cos_theta)
        // This matches: cos_theta = midpoints of linspace(1, -1, theta_bins+1)
        Float cos_theta = 1.f - 2.f * center_y;
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

        // No differentials since all rays in a pixel have the same direction
        ray.has_differentials = false;

        return { ray, ray_weight & active };
    }

    // This sensor does not occupy any particular region of space
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "DistantKitten[" << std::endl
            << "  sampling = center," << std::endl
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

MI_EXPORT_PLUGIN(DistantKitten)

NAMESPACE_END(mitsuba)
