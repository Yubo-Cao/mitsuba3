#include <mitsuba/core/bbox.h>
#include <mitsuba/core/distr_1d.h>
#include <mitsuba/core/frame.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/volume.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _emitter-gridemit:

Grid Volumetric Emitter (:monosp:`gridemit`)
--------------------------------------------

.. pluginparameters::

 * - emission
   - |volume|
   - A 3D volume texture specifying the isotropic emission intensity at each point.
     The volume's to_world transform defines the coordinate mapping.
   - |exposed|, |differentiable|

 * - extent
   - |vector|
   - Half-extents of the emitter bounding box in world space.
     The emitter spans from [-extent, extent]³. (Default: [1, 1, 1])
   - |exposed|

 * - resolution
   - |vector| (integer)
   - Resolution for importance sampling distribution. Should match the GridVolume
     resolution for best results. (Default: [32, 32, 32])

This emitter implements a volumetric light source where each voxel emits light
isotropically based on the emission volume intensity.

The emission volume's to_world transform should map [0,1]³ to match the extent box.
For extent=[1,1,1], use to_world: translate([-1,-1,-1]).scale([2,2,2]).

*/

template <typename Float, typename Spectrum>
class VolumetricEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_medium, m_needs_sample_3, m_to_world)
    MI_IMPORT_TYPES(Scene, Shape, Texture, Volume)

    VolumetricEmitter(const Properties &props) : Base(props) {
        // Load the emission volume
        m_emission = props.get_volume<Volume>("emission", 1.f);

        // Half-extents: emitter box spans [-extent, extent]³ in world space
        m_extent = props.get<ScalarVector3f>("extent", ScalarVector3f(1.f, 1.f, 1.f));

        // Resolution for importance sampling
        ScalarVector3i default_res(32, 32, 32);
        if (props.has_property("resolution")) {
            auto res_prop = props.get<ScalarVector3f>("resolution", ScalarVector3f(32.f, 32.f, 32.f));
            m_resolution = ScalarVector3i((int)res_prop.x(), (int)res_prop.y(), (int)res_prop.z());
        } else {
            m_resolution = default_res;
        }

        // Compute derived quantities
        m_bbox = ScalarBoundingBox3f(
            ScalarPoint3f(-m_extent.x(), -m_extent.y(), -m_extent.z()),
            ScalarPoint3f(m_extent.x(), m_extent.y(), m_extent.z())
        );

        m_voxel_size = ScalarVector3f(
            2.f * m_extent.x() / m_resolution.x(),
            2.f * m_extent.y() / m_resolution.y(),
            2.f * m_extent.z() / m_resolution.z()
        );

        m_voxel_volume = m_voxel_size.x() * m_voxel_size.y() * m_voxel_size.z();

        // Store as opaque drjit values to prevent recompilation
        m_extent_x = dr::opaque<Float>(m_extent.x());
        m_extent_y = dr::opaque<Float>(m_extent.y());
        m_extent_z = dr::opaque<Float>(m_extent.z());
        m_voxel_size_x = dr::opaque<Float>(m_voxel_size.x());
        m_voxel_size_y = dr::opaque<Float>(m_voxel_size.y());
        m_voxel_size_z = dr::opaque<Float>(m_voxel_size.z());
        m_voxel_volume_dr = dr::opaque<Float>(m_voxel_volume);
        m_res_x = dr::opaque<UInt32>((uint32_t)m_resolution.x());
        m_res_y = dr::opaque<UInt32>((uint32_t)m_resolution.y());
        m_res_z = dr::opaque<UInt32>((uint32_t)m_resolution.z());

        // Build the importance sampling distribution
        build_distribution();

        m_needs_sample_3 = true;
        m_flags = +EmitterFlags::SpatiallyVarying;
        
        Log(Info, "VolumetricEmitter: extent=[%f,%f,%f], resolution=[%d,%d,%d], total_power=%f",
            m_extent.x(), m_extent.y(), m_extent.z(),
            m_resolution.x(), m_resolution.y(), m_resolution.z(),
            m_total_power);
    }

    void traverse(TraversalCallback *cb) override {
        Base::traverse(cb);
        cb->put("emission", m_emission, ParamFlags::Differentiable);
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        if (keys.empty() || string::contains(keys, "emission")) {
            build_distribution();
        }
        Base::parameters_changed(keys);
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &pos_sample,
                                          const Point2f &dir_sample,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // Sample wavelengths
        auto [wavelengths, weight] = sample_wavelengths(
            dr::zeros<SurfaceInteraction3f>(), wavelength_sample, active);

        // Sample a position in the volume (importance sampled by emission)
        auto [position, pos_pdf] = sample_position_impl(pos_sample, active);

        // Evaluate actual emission at sampled point
        Interaction3f it;
        it.p = position;
        UnpolarizedSpectrum emission_val = m_emission->eval(it, active);

        // Sample direction uniformly on sphere
        Vector3f direction = warp::square_to_uniform_sphere(dir_sample);

        // Create ray (offset slightly to avoid self-intersection)
        Ray3f ray(position + direction * math::RayEpsilon<Float>, direction, time, wavelengths);

        // Weight: emission * 4π / pdf
        // pdf = (emission / total_emission) / voxel_volume
        // So weight = total_emission * voxel_volume * 4π / voxel_volume = total_emission * 4π
        // But we want emission-weighted, so: emission / pdf * 4π = total * 4π when properly normalized
        Float inv_pdf = dr::select(pos_pdf > 0, dr::rcp(pos_pdf), 0.f);
        weight *= emission_val * inv_pdf * 4.f * dr::Pi<Float>;

        return { ray, depolarizer<Spectrum>(weight) & active };
    }

    std::pair<DirectionSample3f, Spectrum> sample_direction(const Interaction3f &it,
                                                            const Point2f &sample,
                                                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        // Sample a position in the volume
        auto [position, pos_pdf] = sample_position_impl(sample, active);

        DirectionSample3f ds;
        ds.p       = position;
        ds.n       = 0.f;
        ds.uv      = 0.f;
        ds.time    = it.time;
        ds.delta   = false;
        ds.emitter = this;
        ds.d       = ds.p - it.p;

        Float dist2    = dr::squared_norm(ds.d);
        Float dist     = dr::sqrt(dist2);
        Float inv_dist = dr::rcp(dist);
        ds.dist = dist;
        ds.d *= inv_dist;

        // PDF in solid angle measure: pos_pdf * dist² (for isotropic emission, no cos factor)
        // Divided by 4π because emission is isotropic over the sphere
        ds.pdf = pos_pdf * dist2 * dr::rcp(4.f * dr::Pi<Float>);

        // Evaluate emission at the sampled position
        Interaction3f mi;
        mi.p = position;
        UnpolarizedSpectrum emission_value = m_emission->eval(mi, active);

        // Radiance: emission / (4π) at unit distance, then /dist² for falloff
        // L = emission / (4π * dist²)
        UnpolarizedSpectrum spec = emission_value * dr::rcp(4.f * dr::Pi<Float> * dist2);

        return { ds, depolarizer<Spectrum>(spec) & active };
    }

    Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
                        Mask /* active */) const override {
        // Get position PDF at ds.p
        Float pos_pdf = pdf_position_impl(ds.p);
        Float dist2 = dr::squared_norm(ds.p - it.p);
        
        // Convert to solid angle measure
        return pos_pdf * dist2 * dr::rcp(4.f * dr::Pi<Float>);
    }

    Spectrum eval_direction(const Interaction3f &it,
                            const DirectionSample3f &ds,
                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        // Check if point is inside bounding box
        Mask inside = m_bbox.contains(ds.p);
        active &= inside;

        // Evaluate emission at the position
        Interaction3f mi;
        mi.p = ds.p;
        UnpolarizedSpectrum emission_value = m_emission->eval(mi, active);

        // Isotropic emission with geometric falloff
        Float dist2 = dr::squared_norm(ds.p - it.p);
        UnpolarizedSpectrum spec = emission_value * dr::rcp(4.f * dr::Pi<Float> * dist2);

        return depolarizer<Spectrum>(spec) & active;
    }

    std::pair<PositionSample3f, Float>
    sample_position(Float time, const Point2f &sample,
                    Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSamplePosition, active);

        auto [position, pdf] = sample_position_impl(sample, active);

        PositionSample3f ps = dr::zeros<PositionSample3f>();
        ps.p = position;
        ps.time = time;
        ps.delta = false;

        return { ps, pdf };
    }

    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f & /* si */, Float sample,
                       Mask /* active */) const override {
        Wavelength wavelengths = math::sample_shifted<Wavelength>(sample);
        return { wavelengths, Spectrum(1.f) };
    }

    Spectrum eval(const SurfaceInteraction3f &, Mask) const override {
        return 0.f;
    }

    ScalarBoundingBox3f bbox() const override {
        return m_bbox;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "VolumetricEmitter[" << std::endl
            << "  emission = " << string::indent(m_emission) << "," << std::endl
            << "  extent = " << m_extent << "," << std::endl
            << "  resolution = " << m_resolution << "," << std::endl
            << "  total_power = " << m_total_power << "," << std::endl
            << "  medium = " << (m_medium ? string::indent(m_medium) : "none")
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(VolumetricEmitter)

private:
    /// Build the importance sampling distribution by evaluating the volume at voxel centers
    void build_distribution() {
        size_t total_voxels = (size_t)m_resolution.x() * m_resolution.y() * m_resolution.z();
        
        // Build arrays of all voxel center positions
        std::vector<ScalarFloat> px(total_voxels), py(total_voxels), pz(total_voxels);
        
        for (int iz = 0; iz < m_resolution.z(); ++iz) {
            for (int iy = 0; iy < m_resolution.y(); ++iy) {
                for (int ix = 0; ix < m_resolution.x(); ++ix) {
                    ScalarFloat tx = (ScalarFloat(ix) + 0.5f) / ScalarFloat(m_resolution.x());
                    ScalarFloat ty = (ScalarFloat(iy) + 0.5f) / ScalarFloat(m_resolution.y());
                    ScalarFloat tz = (ScalarFloat(iz) + 0.5f) / ScalarFloat(m_resolution.z());
                    
                    size_t idx = (size_t)iz * m_resolution.y() * m_resolution.x() + 
                                 (size_t)iy * m_resolution.x() + (size_t)ix;
                    
                    px[idx] = -m_extent.x() + tx * 2.f * m_extent.x();
                    py[idx] = -m_extent.y() + ty * 2.f * m_extent.y();
                    pz[idx] = -m_extent.z() + tz * 2.f * m_extent.z();
                }
            }
        }
        
        // Create single vectorized interaction with ALL points - ONE kernel compilation
        Interaction3f it;
        it.p = Point3f(
            dr::load<Float>(px.data(), total_voxels),
            dr::load<Float>(py.data(), total_voxels),
            dr::load<Float>(pz.data(), total_voxels)
        );
        
        // Single batched evaluation of the volume
        auto emission_val = m_emission->eval(it, true);
        
        // Compute luminance/intensity
        Float luminance;
        if constexpr (is_rgb_v<Spectrum>) {
            luminance = 0.212671f * emission_val[0] + 
                        0.715160f * emission_val[1] + 
                        0.072169f * emission_val[2];
        } else {
            luminance = dr::mean(emission_val);
        }
        
        // Ensure non-negative
        luminance = dr::maximum(luminance, 0.f);
        
        // Force evaluation and extract to CPU
        dr::eval(luminance);
        dr::sync_thread();
        
        std::vector<ScalarFloat> weights(total_voxels);
        dr::store(weights.data(), luminance);
        
        // Compute statistics
        ScalarFloat total = 0;
        ScalarFloat max_val = 0;
        size_t nonzero_count = 0;
        
        for (size_t i = 0; i < total_voxels; ++i) {
            ScalarFloat w = weights[i];
            if (w < 0) w = 0;
            weights[i] = w;
            total += w;
            if (w > max_val) max_val = w;
            if (w > 0) nonzero_count++;
        }

        Log(Info, "VolumetricEmitter distribution: total=%f, max=%f, nonzero=%zu/%zu",
            total, max_val, nonzero_count, total_voxels);

        // Store total power (integrated emission over volume)
        m_total_power = total * m_voxel_volume;

        // Normalize weights for the distribution
        if (total > 0) {
            for (auto &w : weights)
                w /= total;
        } else {
            Log(Warn, "VolumetricEmitter: All emission weights are zero!");
            ScalarFloat uniform = ScalarFloat(1) / ScalarFloat(total_voxels);
            for (auto &w : weights)
                w = uniform;
        }

        // Build discrete distribution with Float type for vectorized sampling
        m_distr = DiscreteDistribution<Float>(weights.data(), total_voxels);
    }

    /// Sample a position using importance sampling
    std::pair<Point3f, Float> sample_position_impl(const Point2f &sample, Mask active) const {
        // Sample voxel index from discrete distribution
        Float sample_x = sample.x();
        auto [voxel_idx, voxel_pdf] = m_distr.sample_reuse(sample_x, active);

        // Convert 1D index to 3D indices (z-major order)
        UInt32 idx = UInt32(voxel_idx);
        UInt32 res_xy = m_res_x * m_res_y;
        UInt32 iz = idx / res_xy;
        UInt32 rem = idx % res_xy;
        UInt32 iy = rem / m_res_x;
        UInt32 ix = rem % m_res_x;

        // Random position within voxel
        Float u = sample_x;  // Reused from discrete sampling
        Float v = sample.y();
        Float w_raw = sample.y() + 0.5f * sample_x + 0.38196601125f;
        Float w = w_raw - dr::floor(w_raw);

        // Compute world position
        // (ix + u) / res_x gives [0,1], then map to [-extent, extent]
        Float tx = (Float(ix) + u) / Float(m_res_x);
        Float ty = (Float(iy) + v) / Float(m_res_y);
        Float tz = (Float(iz) + w) / Float(m_res_z);

        Point3f position(
            -m_extent_x + tx * 2.f * m_extent_x,
            -m_extent_y + ty * 2.f * m_extent_y,
            -m_extent_z + tz * 2.f * m_extent_z
        );

        // PDF is voxel_pdf / voxel_volume
        Float pdf = voxel_pdf / m_voxel_volume_dr;

        return { position, pdf };
    }

    /// Compute PDF for a given position
    Float pdf_position_impl(const Point3f &p) const {
        // Map world position to [0,1]³
        Float tx = (p.x() + m_extent_x) / (2.f * m_extent_x);
        Float ty = (p.y() + m_extent_y) / (2.f * m_extent_y);
        Float tz = (p.z() + m_extent_z) / (2.f * m_extent_z);

        // Get voxel indices
        Int32 ix = dr::clip(Int32(dr::floor(tx * Float(m_res_x))), 0, Int32(m_res_x) - 1);
        Int32 iy = dr::clip(Int32(dr::floor(ty * Float(m_res_y))), 0, Int32(m_res_y) - 1);
        Int32 iz = dr::clip(Int32(dr::floor(tz * Float(m_res_z))), 0, Int32(m_res_z) - 1);

        UInt32 idx = UInt32(iz) * m_res_x * m_res_y + UInt32(iy) * m_res_x + UInt32(ix);

        // Get voxel PDF from distribution
        Float voxel_pdf = m_distr.eval_pmf_normalized(idx);

        return voxel_pdf / m_voxel_volume_dr;
    }

    ref<Volume> m_emission;
    ScalarVector3f m_extent;
    ScalarVector3i m_resolution;
    ScalarBoundingBox3f m_bbox;
    ScalarVector3f m_voxel_size;
    ScalarFloat m_voxel_volume;
    ScalarFloat m_total_power;
    DiscreteDistribution<Float> m_distr;
    
    // Opaque drjit values to prevent kernel recompilation
    Float m_extent_x, m_extent_y, m_extent_z;
    Float m_voxel_size_x, m_voxel_size_y, m_voxel_size_z;
    Float m_voxel_volume_dr;
    UInt32 m_res_x, m_res_y, m_res_z;

    MI_TRAVERSE_CB(Base, m_emission)
};

MI_EXPORT_PLUGIN(VolumetricEmitter)
NAMESPACE_END(mitsuba)