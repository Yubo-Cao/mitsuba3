/**
 * Spherical Basis Emitter for Mitsuba 3
 *
 * A central directional emitter that evaluates analytical basis functions
 * (Gaussian, Heaviside) directly without rasterization artifacts.
 *
 * This emitter acts as an infinitely distant light source with directional
 * variation defined by a sum of continuous basis functions on the sphere.
 *
 * Coordinate convention (physics convention, matching distantmeow sensor):
 *   - theta: polar angle from +z axis [0, π]
 *   - phi: azimuthal angle from +x axis [0, 2π)
 *   - omega = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
 */

#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/distr_2d.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/texture.h>
#include <sstream>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-sphericalbasis:

Spherical Basis Emitter (:monosp:`sphericalbasis`)
--------------------------------------------------

.. pluginparameters::

 * - gaussian_params
   - |string|
   - Semicolon-separated Gaussian basis functions. Each basis is specified as:
     "mu_x,mu_y,mu_z,sigma,intensity". Example: "0,0,1,0.1,1.0;1,0,0,0.2,0.5"

 * - heaviside_params
   - |string|
   - Semicolon-separated Heaviside basis functions. Each basis is specified as:
     "theta_min,theta_max,phi_min,phi_max,intensity". Example: "0,0.5,0,3.14,1.0"

 * - clamp_min
   - |float|
   - Minimum clamp value (default: 0)

 * - clamp_max
   - |float|
   - Maximum clamp value (default: infinity)

 * - sampling_resolution
   - |int|
   - Resolution for importance sampling grid (default: 64)

This emitter implements a spherical light source centered at a point, where
the radiance varies by direction according to a sum of analytical basis functions.

Unlike bitmap-based emitters, this evaluates the basis functions continuously,
avoiding rasterization artifacts and reducing memory usage.

*/

// Helper function to parse comma-separated floats
static std::vector<double> parse_floats(const std::string &s) {
    std::vector<double> result;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        result.push_back(std::stod(token));
    }
    return result;
}

template <typename Float, typename Spectrum>
class SphericalBasisEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_to_world, m_needs_sample_3)
    MI_IMPORT_TYPES(Scene, Texture)

    using Warp = Hierarchical2D<Float, 0>;

    SphericalBasisEmitter(const Properties &props) : Base(props) {
        // Initialize bounding sphere (updated in set_scene)
        m_bsphere = ScalarBoundingSphere3f(ScalarPoint3f(0.f), 1.f);

        // Sampling resolution
        m_sampling_res = props.get<int>("sampling_resolution", 64);

        // Clamp range
        m_clamp_min = props.get<ScalarFloat>("clamp_min", 0.f);
        m_clamp_max = props.get<ScalarFloat>("clamp_max", std::numeric_limits<ScalarFloat>::infinity());

        // Parse Gaussian bases from string: "mu_x,mu_y,mu_z,sigma,intensity;..."
        std::string gaussian_str = props.get<std::string>("gaussian_params", "");
        if (!gaussian_str.empty()) {
            std::stringstream ss(gaussian_str);
            std::string basis_str;
            while (std::getline(ss, basis_str, ';')) {
                if (basis_str.empty()) continue;
                auto vals = parse_floats(basis_str);
                if (vals.size() != 5) {
                    Throw("Each Gaussian basis must have 5 values: mu_x,mu_y,mu_z,sigma,intensity");
                }
                m_gaussian_mu_x.push_back((ScalarFloat)vals[0]);
                m_gaussian_mu_y.push_back((ScalarFloat)vals[1]);
                m_gaussian_mu_z.push_back((ScalarFloat)vals[2]);
                m_gaussian_sigma.push_back((ScalarFloat)vals[3]);
                m_gaussian_intensity.push_back((ScalarFloat)vals[4]);
            }
        }
        m_n_gaussian = (int)m_gaussian_mu_x.size();

        // Parse Heaviside bases from string: "theta_min,theta_max,phi_min,phi_max,intensity;..."
        std::string heaviside_str = props.get<std::string>("heaviside_params", "");
        if (!heaviside_str.empty()) {
            std::stringstream ss(heaviside_str);
            std::string basis_str;
            while (std::getline(ss, basis_str, ';')) {
                if (basis_str.empty()) continue;
                auto vals = parse_floats(basis_str);
                if (vals.size() != 5) {
                    Throw("Each Heaviside basis must have 5 values: theta_min,theta_max,phi_min,phi_max,intensity");
                }
                m_heaviside_theta_min.push_back((ScalarFloat)vals[0]);
                m_heaviside_theta_max.push_back((ScalarFloat)vals[1]);
                m_heaviside_phi_min.push_back((ScalarFloat)vals[2]);
                m_heaviside_phi_max.push_back((ScalarFloat)vals[3]);
                m_heaviside_intensity.push_back((ScalarFloat)vals[4]);
            }
        }
        m_n_heaviside = (int)m_heaviside_theta_min.size();

        if (m_n_gaussian == 0 && m_n_heaviside == 0)
            Log(Warn, "SphericalBasisEmitter: No basis functions specified!");

        // Build importance sampling distribution
        build_distribution();

        m_needs_sample_3 = false;
        m_flags = EmitterFlags::Infinite | EmitterFlags::SpatiallyVarying;

        Log(Info, "SphericalBasisEmitter: %d Gaussian + %d Heaviside bases, "
                  "sampling_res=%d, clamp=[%f, %f]",
            m_n_gaussian, m_n_heaviside, m_sampling_res, m_clamp_min, m_clamp_max);
    }

    void traverse(TraversalCallback *cb) override {
        Base::traverse(cb);
        cb->put("to_world", m_to_world, ParamFlags::NonDifferentiable);
    }

    void set_scene(const Scene *scene) override {
        if (scene->bbox().valid()) {
            m_bsphere = scene->bbox().bounding_sphere();
            m_bsphere.radius =
                dr::maximum(math::RayEpsilon<Float>,
                            m_bsphere.radius * (1.f + math::RayEpsilon<Float>));
        } else {
            m_bsphere.center = 0.f;
            m_bsphere.radius = math::RayEpsilon<Float>;
        }
    }

    /// Evaluate the sum of all basis functions at a given direction (in local frame)
    Float eval_bases(const Vector3f &omega, Mask active) const {
        Float result = 0.f;

        // Evaluate Gaussian bases: f(omega) = intensity * exp(-0.5 * (angle/sigma)^2)
        for (int i = 0; i < m_n_gaussian; ++i) {
            Vector3f mu(m_gaussian_mu_x[i], m_gaussian_mu_y[i], m_gaussian_mu_z[i]);
            Float dot_val = dr::dot(omega, mu);
            dot_val = dr::clip(dot_val, -1.f, 1.f);
            Float angle = dr::acos(dot_val);
            Float sigma = m_gaussian_sigma[i];
            Float intensity = m_gaussian_intensity[i];
            result += intensity * dr::exp(-0.5f * dr::square(angle / sigma));
        }

        // Evaluate Heaviside bases: f(theta, phi) = intensity if in range, else 0
        // First compute theta and phi from omega
        Float cos_theta = omega.z();
        Float theta = dr::acos(dr::clip(cos_theta, -1.f, 1.f));
        Float phi = dr::atan2(omega.y(), omega.x());
        phi = dr::select(phi < 0.f, phi + 2.f * dr::Pi<Float>, phi);

        for (int i = 0; i < m_n_heaviside; ++i) {
            Float theta_min = m_heaviside_theta_min[i];
            Float theta_max = m_heaviside_theta_max[i];
            Float phi_min = m_heaviside_phi_min[i];
            Float phi_max = m_heaviside_phi_max[i];
            Float intensity = m_heaviside_intensity[i];

            Mask in_theta = (theta >= theta_min) & (theta < theta_max);
            Mask in_phi = (phi >= phi_min) & (phi < phi_max);
            result += dr::select(in_theta & in_phi & active, intensity, 0.f);
        }

        // Apply clamping
        result = dr::clip(result, m_clamp_min, m_clamp_max);

        return result;
    }

    /// Convert UV coordinates [0,1]² to direction vector
    Vector3f uv_to_direction(const Point2f &uv) const {
        // uv.x -> phi [0, 2π), uv.y -> theta [0, π]
        Float phi = uv.x() * 2.f * dr::Pi<Float>;
        Float theta = uv.y() * dr::Pi<Float>;

        Float sin_theta = dr::sin(theta);
        Float cos_theta = dr::cos(theta);
        auto [sin_phi, cos_phi] = dr::sincos(phi);

        return Vector3f(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
    }

    /// Convert direction to UV coordinates
    Point2f direction_to_uv(const Vector3f &d) const {
        Float theta = dr::acos(dr::clip(d.z(), -1.f, 1.f));
        Float phi = dr::atan2(d.y(), d.x());
        phi = dr::select(phi < 0.f, phi + 2.f * dr::Pi<Float>, phi);

        return Point2f(phi * dr::InvTwoPi<Float>, theta * dr::InvPi<Float>);
    }

    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        // Get direction in local frame
        // For infinite emitters, si.wi is the ray direction; we evaluate at this direction
        // to match the distantmeow sensor's coordinate convention
        Vector3f d = m_to_world.value().inverse() * si.wi;

        Float radiance = eval_bases(d, active);

        return depolarizer<Spectrum>(radiance) & active;
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &spatial_sample,
                                          const Point2f &dir_sample,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // 1. Sample spatial component (disk perpendicular to direction)
        Point2f offset = warp::square_to_uniform_disk_concentric(spatial_sample);

        // 2. Sample directional component using importance sampling
        auto [uv, pdf] = m_warp.sample(dir_sample, nullptr, active);
        active &= pdf > 0.f;

        // Convert UV to direction (in local frame)
        Vector3f d_local = uv_to_direction(uv);

        // Account for sin(theta) in spherical measure
        // Use safe_rsqrt pattern from envmap.cpp
        Float sin_theta_sq = dr::maximum(1.f - dr::square(d_local.z()), dr::square(dr::Epsilon<Float>));
        Float inv_sin_theta = dr::rsqrt(sin_theta_sq);
        pdf *= inv_sin_theta * dr::InvTwoPi<Float> * dr::InvPi<Float>;

        // Transform to world frame (negate for ray direction into scene)
        Vector3f d_global = m_to_world.value() * (-d_local);

        // Compute ray origin (behind the scene, shooting toward center)
        Vector3f perp_offset = Frame3f(-d_global).to_world(Vector3f(offset.x(), offset.y(), 0.f));
        Point3f origin = m_bsphere.center + (perp_offset - d_global) * m_bsphere.radius;

        // 3. Sample wavelengths
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.time = time;
        si.p = origin;
        auto [wavelengths, wav_weight] = sample_wavelengths(si, wavelength_sample, active);

        // Compute weight
        Float radiance = eval_bases(d_local, active);
        Float r2 = dr::square(m_bsphere.radius);
        Spectrum weight = wav_weight * radiance * dr::Pi<Float> * r2 / pdf;

        Ray3f ray(origin, d_global, time, wavelengths);

        return { ray, depolarizer<Spectrum>(weight) & active };
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample,
                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        // Sample direction using importance sampling
        auto [uv, pdf] = m_warp.sample(sample, nullptr, active);
        active &= pdf > 0.f;

        // Convert UV to direction in local frame
        Vector3f d_local = uv_to_direction(uv);

        // Account for sin(theta) in solid angle measure
        Float sin_theta_sq = dr::maximum(1.f - dr::square(d_local.z()), dr::square(dr::Epsilon<Float>));
        Float inv_sin_theta = dr::rsqrt(sin_theta_sq);
        pdf *= inv_sin_theta * dr::InvTwoPi<Float> * dr::InvPi<Float>;

        // Transform to world frame
        Vector3f d = m_to_world.value() * d_local;

        // Compute distance (2 * radius ensures we're outside the scene)
        Float radius = dr::maximum(m_bsphere.radius, dr::norm(it.p - m_bsphere.center));
        Float dist = 2.f * radius;

        DirectionSample3f ds;
        ds.p = it.p + d * dist;
        ds.n = -d;
        ds.uv = uv;
        ds.time = it.time;
        ds.pdf = dr::select(active, pdf, 0.f);
        ds.delta = false;
        ds.emitter = this;
        ds.d = d;
        ds.dist = dist;

        // Evaluate radiance at this direction
        Float radiance = eval_bases(d_local, active);
        Spectrum weight = depolarizer<Spectrum>(radiance) / ds.pdf;

        return { ds, weight & active };
    }

    Float pdf_direction(const Interaction3f & /*it*/, const DirectionSample3f &ds,
                        Mask /*active*/) const override {
        // Get direction in local frame
        Vector3f d = m_to_world.value().inverse() * ds.d;

        // Convert to UV
        Point2f uv = direction_to_uv(d);

        // Get PDF from warp
        Float sin_theta_sq = dr::maximum(1.f - dr::square(d.z()), dr::square(dr::Epsilon<Float>));
        Float inv_sin_theta = dr::rsqrt(sin_theta_sq);

        return m_warp.eval(uv) * inv_sin_theta * dr::InvTwoPi<Float> * dr::InvPi<Float>;
    }

    Spectrum eval_direction(const Interaction3f & /*it*/,
                            const DirectionSample3f &ds,
                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        // Get direction in local frame
        Vector3f d = m_to_world.value().inverse() * ds.d;

        Float radiance = eval_bases(d, active);

        return depolarizer<Spectrum>(radiance) & active;
    }

    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f & /*si*/, Float sample,
                       Mask /*active*/) const override {
        Wavelength wavelengths = math::sample_shifted<Wavelength>(sample);
        return { wavelengths, Spectrum(1.f) };
    }

    std::pair<PositionSample3f, Float>
    sample_position(Float /*time*/, const Point2f & /*sample*/,
                    Mask /*active*/) const override {
        if constexpr (dr::is_jit_v<Float>) {
            return { dr::zeros<PositionSample3f>(), dr::NaN<Float> };
        } else {
            NotImplementedError("sample_position");
        }
    }

    ScalarBoundingBox3f bbox() const override {
        // Infinite emitter has no finite bounding box
        return ScalarBoundingBox3f();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SphericalBasisEmitter[" << std::endl
            << "  n_gaussian = " << m_n_gaussian << "," << std::endl
            << "  n_heaviside = " << m_n_heaviside << "," << std::endl
            << "  sampling_res = " << m_sampling_res << "," << std::endl
            << "  clamp = [" << m_clamp_min << ", " << m_clamp_max << "]," << std::endl
            << "  bsphere = " << string::indent(m_bsphere) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()

private:
    /// Build the hierarchical importance sampling distribution
    void build_distribution() {
        size_t res = (size_t)m_sampling_res;
        size_t total = res * res;

        std::unique_ptr<ScalarFloat[]> weights(new ScalarFloat[total]);

        ScalarFloat theta_scale = ScalarFloat(dr::Pi<double> / res);
        ScalarFloat phi_scale = ScalarFloat(2.0 * dr::Pi<double> / res);

        for (size_t y = 0; y < res; ++y) {
            ScalarFloat theta = (ScalarFloat(y) + 0.5f) * theta_scale;
            ScalarFloat sin_theta = std::sin(theta);
            ScalarFloat cos_theta = std::cos(theta);

            for (size_t x = 0; x < res; ++x) {
                ScalarFloat phi = (ScalarFloat(x) + 0.5f) * phi_scale;
                ScalarFloat sin_phi = std::sin(phi);
                ScalarFloat cos_phi = std::cos(phi);

                // Direction vector
                ScalarVector3f omega(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);

                // Evaluate bases at this direction
                ScalarFloat val = 0.f;

                // Gaussian bases
                for (int i = 0; i < m_n_gaussian; ++i) {
                    ScalarVector3f mu(m_gaussian_mu_x[i], m_gaussian_mu_y[i], m_gaussian_mu_z[i]);
                    ScalarFloat dot_val = dr::dot(omega, mu);
                    dot_val = std::max(ScalarFloat(-1), std::min(ScalarFloat(1), dot_val));
                    ScalarFloat angle = std::acos(dot_val);
                    ScalarFloat sigma = m_gaussian_sigma[i];
                    ScalarFloat intensity = m_gaussian_intensity[i];
                    val += intensity * std::exp(-0.5f * (angle / sigma) * (angle / sigma));
                }

                // Heaviside bases
                for (int i = 0; i < m_n_heaviside; ++i) {
                    bool in_theta = (theta >= m_heaviside_theta_min[i]) &&
                                    (theta < m_heaviside_theta_max[i]);
                    bool in_phi = (phi >= m_heaviside_phi_min[i]) &&
                                  (phi < m_heaviside_phi_max[i]);
                    if (in_theta && in_phi)
                        val += m_heaviside_intensity[i];
                }

                // Apply clamping
                val = std::max(m_clamp_min, std::min(m_clamp_max, val));

                // Weight by sin(theta) for proper spherical measure
                weights[y * res + x] = val * sin_theta;
            }
        }

        // Create hierarchical warp for importance sampling
        m_warp = Warp(weights.get(), ScalarVector2u((uint32_t)res, (uint32_t)res));

        Log(Info, "SphericalBasisEmitter: Built importance sampling distribution (%zux%zu)", res, res);
    }

    // Bounding sphere for the scene
    ScalarBoundingSphere3f m_bsphere;

    // Importance sampling
    int m_sampling_res;
    Warp m_warp;

    // Clamping
    ScalarFloat m_clamp_min, m_clamp_max;

    // Gaussian basis parameters
    int m_n_gaussian;
    std::vector<ScalarFloat> m_gaussian_mu_x, m_gaussian_mu_y, m_gaussian_mu_z;
    std::vector<ScalarFloat> m_gaussian_sigma;
    std::vector<ScalarFloat> m_gaussian_intensity;

    // Heaviside basis parameters
    int m_n_heaviside;
    std::vector<ScalarFloat> m_heaviside_theta_min, m_heaviside_theta_max;
    std::vector<ScalarFloat> m_heaviside_phi_min, m_heaviside_phi_max;
    std::vector<ScalarFloat> m_heaviside_intensity;
};

MI_EXPORT_PLUGIN(SphericalBasisEmitter)
NAMESPACE_END(mitsuba)
