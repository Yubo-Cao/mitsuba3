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
 *
 * Parameters are stored as Dr.Jit arrays and made opaque to prevent kernel
 * recompilation when values change. Uses dr::while_loop for dynamic iteration
 * so that even changing the number of bases doesn't require recompilation.
 */

#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/distr_2d.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/texture.h>
#include <drjit/while_loop.h>
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

All basis function parameters are stored as Dr.Jit arrays and can be updated
at runtime via mi.traverse() without triggering kernel recompilation.

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
    using FloatStorage = DynamicBuffer<Float>;
    using UInt32Storage = DynamicBuffer<UInt32>;

    SphericalBasisEmitter(const Properties &props) : Base(props) {
        // Initialize bounding sphere (updated in set_scene)
        m_bsphere = ScalarBoundingSphere3f(ScalarPoint3f(0.f), 1.f);

        // Sampling resolution
        m_sampling_res = props.get<int>("sampling_resolution", 64);

        // Clamp range
        m_clamp_min = props.get<ScalarFloat>("clamp_min", 0.f);
        m_clamp_max = props.get<ScalarFloat>("clamp_max", std::numeric_limits<ScalarFloat>::infinity());

        // Temporary vectors for parsing
        std::vector<ScalarFloat> gauss_mu_x, gauss_mu_y, gauss_mu_z;
        std::vector<ScalarFloat> gauss_sigma, gauss_intensity;
        std::vector<ScalarFloat> heavi_theta_min, heavi_theta_max;
        std::vector<ScalarFloat> heavi_phi_min, heavi_phi_max, heavi_intensity;

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
                gauss_mu_x.push_back((ScalarFloat)vals[0]);
                gauss_mu_y.push_back((ScalarFloat)vals[1]);
                gauss_mu_z.push_back((ScalarFloat)vals[2]);
                gauss_sigma.push_back((ScalarFloat)vals[3]);
                gauss_intensity.push_back((ScalarFloat)vals[4]);
            }
        }
        m_n_gaussian_scalar = (uint32_t)gauss_mu_x.size();

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
                heavi_theta_min.push_back((ScalarFloat)vals[0]);
                heavi_theta_max.push_back((ScalarFloat)vals[1]);
                heavi_phi_min.push_back((ScalarFloat)vals[2]);
                heavi_phi_max.push_back((ScalarFloat)vals[3]);
                heavi_intensity.push_back((ScalarFloat)vals[4]);
            }
        }
        m_n_heaviside_scalar = (uint32_t)heavi_theta_min.size();

        if (m_n_gaussian_scalar == 0 && m_n_heaviside_scalar == 0)
            Log(Warn, "SphericalBasisEmitter: No basis functions specified!");

        // Convert to Dr.Jit arrays for opaque parameter updates
        init_drjit_arrays(gauss_mu_x, gauss_mu_y, gauss_mu_z, gauss_sigma, gauss_intensity,
                          heavi_theta_min, heavi_theta_max, heavi_phi_min, heavi_phi_max, heavi_intensity);

        // Build importance sampling distribution
        build_distribution();

        m_needs_sample_3 = false;
        m_flags = EmitterFlags::Infinite | EmitterFlags::SpatiallyVarying;

        Log(Info, "SphericalBasisEmitter: %u Gaussian + %u Heaviside bases, "
                  "sampling_res=%d, clamp=[%f, %f]",
            m_n_gaussian_scalar, m_n_heaviside_scalar, m_sampling_res, m_clamp_min, m_clamp_max);
    }

    void init_drjit_arrays(
        const std::vector<ScalarFloat> &gauss_mu_x,
        const std::vector<ScalarFloat> &gauss_mu_y,
        const std::vector<ScalarFloat> &gauss_mu_z,
        const std::vector<ScalarFloat> &gauss_sigma,
        const std::vector<ScalarFloat> &gauss_intensity,
        const std::vector<ScalarFloat> &heavi_theta_min,
        const std::vector<ScalarFloat> &heavi_theta_max,
        const std::vector<ScalarFloat> &heavi_phi_min,
        const std::vector<ScalarFloat> &heavi_phi_max,
        const std::vector<ScalarFloat> &heavi_intensity
    ) {
        // Store counts as opaque Dr.Jit values to prevent recompilation
        m_n_gaussian = UInt32(m_n_gaussian_scalar);
        m_n_heaviside = UInt32(m_n_heaviside_scalar);
        dr::make_opaque(m_n_gaussian, m_n_heaviside);

        // Initialize Gaussian parameter arrays
        if (m_n_gaussian_scalar > 0) {
            m_gaussian_mu_x = dr::load<FloatStorage>(gauss_mu_x.data(), m_n_gaussian_scalar);
            m_gaussian_mu_y = dr::load<FloatStorage>(gauss_mu_y.data(), m_n_gaussian_scalar);
            m_gaussian_mu_z = dr::load<FloatStorage>(gauss_mu_z.data(), m_n_gaussian_scalar);
            m_gaussian_sigma = dr::load<FloatStorage>(gauss_sigma.data(), m_n_gaussian_scalar);
            m_gaussian_intensity = dr::load<FloatStorage>(gauss_intensity.data(), m_n_gaussian_scalar);

            dr::make_opaque(m_gaussian_mu_x, m_gaussian_mu_y, m_gaussian_mu_z,
                           m_gaussian_sigma, m_gaussian_intensity);
        } else {
            // Initialize with dummy single-element arrays for consistent kernel structure
            m_gaussian_mu_x = dr::zeros<FloatStorage>(1);
            m_gaussian_mu_y = dr::zeros<FloatStorage>(1);
            m_gaussian_mu_z = dr::zeros<FloatStorage>(1);
            m_gaussian_sigma = dr::full<FloatStorage>(1.f, 1);
            m_gaussian_intensity = dr::zeros<FloatStorage>(1);
            dr::make_opaque(m_gaussian_mu_x, m_gaussian_mu_y, m_gaussian_mu_z,
                           m_gaussian_sigma, m_gaussian_intensity);
        }

        // Initialize Heaviside parameter arrays
        if (m_n_heaviside_scalar > 0) {
            m_heaviside_theta_min = dr::load<FloatStorage>(heavi_theta_min.data(), m_n_heaviside_scalar);
            m_heaviside_theta_max = dr::load<FloatStorage>(heavi_theta_max.data(), m_n_heaviside_scalar);
            m_heaviside_phi_min = dr::load<FloatStorage>(heavi_phi_min.data(), m_n_heaviside_scalar);
            m_heaviside_phi_max = dr::load<FloatStorage>(heavi_phi_max.data(), m_n_heaviside_scalar);
            m_heaviside_intensity = dr::load<FloatStorage>(heavi_intensity.data(), m_n_heaviside_scalar);

            dr::make_opaque(m_heaviside_theta_min, m_heaviside_theta_max,
                           m_heaviside_phi_min, m_heaviside_phi_max, m_heaviside_intensity);
        } else {
            // Initialize with dummy single-element arrays for consistent kernel structure
            m_heaviside_theta_min = dr::zeros<FloatStorage>(1);
            m_heaviside_theta_max = dr::zeros<FloatStorage>(1);
            m_heaviside_phi_min = dr::zeros<FloatStorage>(1);
            m_heaviside_phi_max = dr::zeros<FloatStorage>(1);
            m_heaviside_intensity = dr::zeros<FloatStorage>(1);
            dr::make_opaque(m_heaviside_theta_min, m_heaviside_theta_max,
                           m_heaviside_phi_min, m_heaviside_phi_max, m_heaviside_intensity);
        }
    }

    void traverse(TraversalCallback *cb) override {
        Base::traverse(cb);
        cb->put("to_world", *m_to_world.ptr(), ParamFlags::NonDifferentiable);

        // Expose basis counts for runtime updates
        cb->put("n_gaussian", m_n_gaussian, ParamFlags::NonDifferentiable);
        cb->put("n_heaviside", m_n_heaviside, ParamFlags::NonDifferentiable);

        // Expose Gaussian parameters for runtime updates
        cb->put("gaussian_mu_x", m_gaussian_mu_x, ParamFlags::NonDifferentiable);
        cb->put("gaussian_mu_y", m_gaussian_mu_y, ParamFlags::NonDifferentiable);
        cb->put("gaussian_mu_z", m_gaussian_mu_z, ParamFlags::NonDifferentiable);
        cb->put("gaussian_sigma", m_gaussian_sigma, ParamFlags::NonDifferentiable);
        cb->put("gaussian_intensity", m_gaussian_intensity, ParamFlags::NonDifferentiable);

        // Expose Heaviside parameters for runtime updates
        cb->put("heaviside_theta_min", m_heaviside_theta_min, ParamFlags::NonDifferentiable);
        cb->put("heaviside_theta_max", m_heaviside_theta_max, ParamFlags::NonDifferentiable);
        cb->put("heaviside_phi_min", m_heaviside_phi_min, ParamFlags::NonDifferentiable);
        cb->put("heaviside_phi_max", m_heaviside_phi_max, ParamFlags::NonDifferentiable);
        cb->put("heaviside_intensity", m_heaviside_intensity, ParamFlags::NonDifferentiable);

        // Expose clamp parameters
        cb->put("clamp_min", m_clamp_min, ParamFlags::NonDifferentiable);
        cb->put("clamp_max", m_clamp_max, ParamFlags::NonDifferentiable);
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        // Update scalar counts from Dr.Jit values
        if (dr::width(m_n_gaussian) > 0)
            m_n_gaussian_scalar = dr::slice(m_n_gaussian, 0);
        if (dr::width(m_n_heaviside) > 0)
            m_n_heaviside_scalar = dr::slice(m_n_heaviside, 0);

        // Make parameters opaque after updates
        dr::make_opaque(m_n_gaussian, m_n_heaviside);
        dr::make_opaque(m_gaussian_mu_x, m_gaussian_mu_y, m_gaussian_mu_z,
                       m_gaussian_sigma, m_gaussian_intensity);
        dr::make_opaque(m_heaviside_theta_min, m_heaviside_theta_max,
                       m_heaviside_phi_min, m_heaviside_phi_max, m_heaviside_intensity);

        // Rebuild importance sampling distribution if parameters changed
        bool rebuild_dist = keys.empty();
        for (const auto &key : keys) {
            if (key.find("gaussian") != std::string::npos ||
                key.find("heaviside") != std::string::npos ||
                key.find("clamp") != std::string::npos ||
                key.find("n_") != std::string::npos) {
                rebuild_dist = true;
                break;
            }
        }

        if (rebuild_dist) {
            build_distribution();
        }
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
    /// Uses dr::while_loop for dynamic iteration to avoid kernel recompilation
    Float eval_bases(const Vector3f &omega, Mask active) const {
        Float result = 0.f;

        // Precompute theta and phi from omega (needed for Heaviside)
        Float cos_theta = omega.z();
        Float theta = dr::acos(dr::clip(cos_theta, -1.f, 1.f));
        Float phi = dr::atan2(omega.y(), omega.x());
        phi = dr::select(phi < 0.f, phi + 2.f * dr::Pi<Float>, phi);

        // Evaluate Gaussian bases using dynamic while_loop
        {
            UInt32 i_init = dr::zeros<UInt32>(dr::width(active));
            Float sum_init = dr::zeros<Float>(dr::width(active));

            // Capture needed values for the loop
            auto [i_final, gauss_sum] = dr::while_loop(
                std::make_tuple(i_init, sum_init),
                // Condition: continue while i < n_gaussian
                [&](const UInt32 &i, const Float &) {
                    return active && (i < m_n_gaussian);
                },
                // Body: accumulate Gaussian contributions
                [&](UInt32 &i, Float &sum) {
                    Float mu_x = dr::gather<Float>(m_gaussian_mu_x, i, active);
                    Float mu_y = dr::gather<Float>(m_gaussian_mu_y, i, active);
                    Float mu_z = dr::gather<Float>(m_gaussian_mu_z, i, active);
                    Float sigma = dr::gather<Float>(m_gaussian_sigma, i, active);
                    Float intensity = dr::gather<Float>(m_gaussian_intensity, i, active);

                    Vector3f mu(mu_x, mu_y, mu_z);
                    Float dot_val = dr::dot(omega, mu);
                    dot_val = dr::clip(dot_val, -1.f, 1.f);
                    Float angle = dr::acos(dot_val);
                    sum += intensity * dr::exp(-0.5f * dr::square(angle / sigma));

                    i += 1u;
                },
                "Gaussian basis eval"
            );
            result += gauss_sum;
        }

        // Evaluate Heaviside bases using dynamic while_loop
        {
            UInt32 i_init = dr::zeros<UInt32>(dr::width(active));
            Float sum_init = dr::zeros<Float>(dr::width(active));

            auto [i_final, heavi_sum] = dr::while_loop(
                std::make_tuple(i_init, sum_init),
                // Condition
                [&](const UInt32 &i, const Float &) {
                    return active && (i < m_n_heaviside);
                },
                // Body
                [&](UInt32 &i, Float &sum) {
                    Float theta_min = dr::gather<Float>(m_heaviside_theta_min, i, active);
                    Float theta_max = dr::gather<Float>(m_heaviside_theta_max, i, active);
                    Float phi_min = dr::gather<Float>(m_heaviside_phi_min, i, active);
                    Float phi_max = dr::gather<Float>(m_heaviside_phi_max, i, active);
                    Float intensity = dr::gather<Float>(m_heaviside_intensity, i, active);

                    Mask in_theta = (theta >= theta_min) & (theta < theta_max);
                    Mask in_phi = (phi >= phi_min) & (phi < phi_max);
                    sum += dr::select(in_theta & in_phi, intensity, 0.f);

                    i += 1u;
                },
                "Heaviside basis eval"
            );
            result += heavi_sum;
        }

        // Apply clamping
        result = dr::clip(result, Float(m_clamp_min), Float(m_clamp_max));

        return result;
    }

    /// Convert UV coordinates [0,1]² to direction vector
    Vector3f uv_to_direction(const Point2f &uv) const {
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

        Vector3f d = m_to_world.value().inverse() * si.wi;
        Float radiance = eval_bases(d, active);

        return depolarizer<Spectrum>(radiance) & active;
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &spatial_sample,
                                          const Point2f &dir_sample,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        Point2f offset = warp::square_to_uniform_disk_concentric(spatial_sample);

        auto [uv, pdf] = m_warp.sample(dir_sample, nullptr, active);
        active &= pdf > 0.f;

        Vector3f d_local = uv_to_direction(uv);

        Float sin_theta_sq = dr::maximum(1.f - dr::square(d_local.z()), dr::square(dr::Epsilon<Float>));
        Float inv_sin_theta = dr::rsqrt(sin_theta_sq);
        pdf *= inv_sin_theta * dr::InvTwoPi<Float> * dr::InvPi<Float>;

        Vector3f d_global = m_to_world.value() * (-d_local);

        Vector3f perp_offset = Frame3f(-d_global).to_world(Vector3f(offset.x(), offset.y(), 0.f));
        Point3f origin = m_bsphere.center + (perp_offset - d_global) * m_bsphere.radius;

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.time = time;
        si.p = origin;
        auto [wavelengths, wav_weight] = sample_wavelengths(si, wavelength_sample, active);

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

        auto [uv, pdf] = m_warp.sample(sample, nullptr, active);
        active &= pdf > 0.f;

        Vector3f d_local = uv_to_direction(uv);

        Float sin_theta_sq = dr::maximum(1.f - dr::square(d_local.z()), dr::square(dr::Epsilon<Float>));
        Float inv_sin_theta = dr::rsqrt(sin_theta_sq);
        pdf *= inv_sin_theta * dr::InvTwoPi<Float> * dr::InvPi<Float>;

        Vector3f d = m_to_world.value() * d_local;

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

        Float radiance = eval_bases(d_local, active);
        Spectrum weight = depolarizer<Spectrum>(radiance) / ds.pdf;

        return { ds, weight & active };
    }

    Float pdf_direction(const Interaction3f & /*it*/, const DirectionSample3f &ds,
                        Mask /*active*/) const override {
        Vector3f d = m_to_world.value().inverse() * ds.d;
        Point2f uv = direction_to_uv(d);

        Float sin_theta_sq = dr::maximum(1.f - dr::square(d.z()), dr::square(dr::Epsilon<Float>));
        Float inv_sin_theta = dr::rsqrt(sin_theta_sq);

        return m_warp.eval(uv) * inv_sin_theta * dr::InvTwoPi<Float> * dr::InvPi<Float>;
    }

    Spectrum eval_direction(const Interaction3f & /*it*/,
                            const DirectionSample3f &ds,
                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

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
        return ScalarBoundingBox3f();
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "SphericalBasisEmitter[" << std::endl
            << "  n_gaussian = " << m_n_gaussian_scalar << "," << std::endl
            << "  n_heaviside = " << m_n_heaviside_scalar << "," << std::endl
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

        // Extract scalar values from Dr.Jit arrays for distribution building
        std::vector<ScalarFloat> gauss_mu_x_s(m_n_gaussian_scalar);
        std::vector<ScalarFloat> gauss_mu_y_s(m_n_gaussian_scalar);
        std::vector<ScalarFloat> gauss_mu_z_s(m_n_gaussian_scalar);
        std::vector<ScalarFloat> gauss_sigma_s(m_n_gaussian_scalar);
        std::vector<ScalarFloat> gauss_intensity_s(m_n_gaussian_scalar);

        std::vector<ScalarFloat> heavi_theta_min_s(m_n_heaviside_scalar);
        std::vector<ScalarFloat> heavi_theta_max_s(m_n_heaviside_scalar);
        std::vector<ScalarFloat> heavi_phi_min_s(m_n_heaviside_scalar);
        std::vector<ScalarFloat> heavi_phi_max_s(m_n_heaviside_scalar);
        std::vector<ScalarFloat> heavi_intensity_s(m_n_heaviside_scalar);

        if (m_n_gaussian_scalar > 0) {
            dr::store(gauss_mu_x_s.data(), m_gaussian_mu_x);
            dr::store(gauss_mu_y_s.data(), m_gaussian_mu_y);
            dr::store(gauss_mu_z_s.data(), m_gaussian_mu_z);
            dr::store(gauss_sigma_s.data(), m_gaussian_sigma);
            dr::store(gauss_intensity_s.data(), m_gaussian_intensity);
        }

        if (m_n_heaviside_scalar > 0) {
            dr::store(heavi_theta_min_s.data(), m_heaviside_theta_min);
            dr::store(heavi_theta_max_s.data(), m_heaviside_theta_max);
            dr::store(heavi_phi_min_s.data(), m_heaviside_phi_min);
            dr::store(heavi_phi_max_s.data(), m_heaviside_phi_max);
            dr::store(heavi_intensity_s.data(), m_heaviside_intensity);
        }

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

                ScalarVector3f omega(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);

                ScalarFloat val = 0.f;

                // Gaussian bases
                for (uint32_t i = 0; i < m_n_gaussian_scalar; ++i) {
                    ScalarVector3f mu(gauss_mu_x_s[i], gauss_mu_y_s[i], gauss_mu_z_s[i]);
                    ScalarFloat dot_val = dr::dot(omega, mu);
                    dot_val = std::max(ScalarFloat(-1), std::min(ScalarFloat(1), dot_val));
                    ScalarFloat angle = std::acos(dot_val);
                    ScalarFloat sigma = gauss_sigma_s[i];
                    ScalarFloat intensity = gauss_intensity_s[i];
                    val += intensity * std::exp(-0.5f * (angle / sigma) * (angle / sigma));
                }

                // Heaviside bases
                for (uint32_t i = 0; i < m_n_heaviside_scalar; ++i) {
                    bool in_theta = (theta >= heavi_theta_min_s[i]) &&
                                    (theta < heavi_theta_max_s[i]);
                    bool in_phi = (phi >= heavi_phi_min_s[i]) &&
                                  (phi < heavi_phi_max_s[i]);
                    if (in_theta && in_phi)
                        val += heavi_intensity_s[i];
                }

                val = std::max(m_clamp_min, std::min(m_clamp_max, val));
                weights[y * res + x] = val * sin_theta;
            }
        }

        m_warp = Warp(weights.get(), ScalarVector2u((uint32_t)res, (uint32_t)res));

        Log(Info, "SphericalBasisEmitter: Built importance sampling distribution (%zux%zu)", res, res);
    }

    ScalarBoundingSphere3f m_bsphere;

    int m_sampling_res;
    Warp m_warp;

    ScalarFloat m_clamp_min, m_clamp_max;

    // Scalar counts for CPU-side operations (distribution building, etc.)
    uint32_t m_n_gaussian_scalar;
    uint32_t m_n_heaviside_scalar;

    // Dr.Jit counts for opaque loop bounds (prevents kernel recompilation)
    UInt32 m_n_gaussian;
    UInt32 m_n_heaviside;

    // Gaussian basis parameters - stored as Dr.Jit arrays for opaque access
    FloatStorage m_gaussian_mu_x, m_gaussian_mu_y, m_gaussian_mu_z;
    FloatStorage m_gaussian_sigma;
    FloatStorage m_gaussian_intensity;

    // Heaviside basis parameters - stored as Dr.Jit arrays for opaque access
    FloatStorage m_heaviside_theta_min, m_heaviside_theta_max;
    FloatStorage m_heaviside_phi_min, m_heaviside_phi_max;
    FloatStorage m_heaviside_intensity;
};

MI_EXPORT_PLUGIN(SphericalBasisEmitter)
NAMESPACE_END(mitsuba)
