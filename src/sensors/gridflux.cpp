#include <mitsuba/core/bbox.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sensor-gridflux:

Grid Flux Meter (:monosp:`gridflux`)
------------------------------------

.. pluginparameters::

 * - grid_min
   - |point|
   - Minimum corner of the measurement grid (Default: (-1, -1, -1))

 * - grid_max
   - |point|
   - Maximum corner of the measurement grid (Default: (1, 1, 1))

 * - full_sphere
   - |bool|
   - If true, sample directions over full sphere (4π). If false, sample
     cosine-weighted hemisphere pointing +z. (Default: true)

 * - medium
   - |medium|
   - Reference to the medium that the grid points are inside. This is REQUIRED
     for rays to properly interact with the medium. Without this, rays won't
     know they're inside a participating medium.

This sensor measures scalar flux at a regular 3D grid of points by shooting
rays in random directions and accumulating incoming radiance.

Film layout:
  - Width  = res_x * res_y  (flattened xy plane)
  - Height = res_z

For a 32³ grid: width=1024, height=32.

After render: `img[:,:,0].reshape(res_z, res_y, res_x)` gives [z,y,x] indexing.

Usage example:
```python
"sensor": {
    "type": "gridflux",
    "grid_min": [-1, -1, -1],
    "grid_max": [1, 1, 1],
    "medium": {"type": "ref", "id": "transport_medium"},  # IMPORTANT!
    "film": {...},
}
```

*/

template <typename Float, typename Spectrum>
class GridFluxMeter final : public Sensor<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Sensor, m_film, m_needs_sample_3, m_medium, sample_wavelengths)
    MI_IMPORT_TYPES()

    GridFluxMeter(const Properties &props) : Base(props) {
        m_grid_min = props.get<ScalarPoint3f>("grid_min", ScalarPoint3f(-1.f, -1.f, -1.f));
        m_grid_max = props.get<ScalarPoint3f>("grid_max", ScalarPoint3f(1.f, 1.f, 1.f));
        m_full_sphere = props.get<bool>("full_sphere", true);
        
        m_grid_size = m_grid_max - m_grid_min;
        
        // Store as opaque to prevent kernel recompilation
        m_grid_min_x = dr::opaque<Float>(m_grid_min.x());
        m_grid_min_y = dr::opaque<Float>(m_grid_min.y());
        m_grid_min_z = dr::opaque<Float>(m_grid_min.z());
        m_grid_size_x = dr::opaque<Float>(m_grid_size.x());
        m_grid_size_y = dr::opaque<Float>(m_grid_size.y());
        m_grid_size_z = dr::opaque<Float>(m_grid_size.z());
        
        if (m_film) {
            ScalarVector2i film_size = m_film->size();
            m_film_width = film_size.x();
            m_film_height = film_size.y();
            
            m_res_xy = (int)std::round(std::sqrt((float)m_film_width));
            m_res_z = m_film_height;
            
            if (m_res_xy * m_res_xy != m_film_width) {
                Log(Warn, "GridFluxMeter: Film width %d is not a perfect square. "
                    "Expected %d for res_xy=%d", m_film_width, m_res_xy * m_res_xy, m_res_xy);
            }
        } else {
            m_res_xy = 16;
            m_res_z = 16;
            m_film_width = 256;
            m_film_height = 16;
        }
        
        m_film_width_dr = dr::opaque<Float>((float)m_film_width);
        m_film_height_dr = dr::opaque<Float>((float)m_film_height);
        m_res_xy_dr = dr::opaque<Float>((float)m_res_xy);
        m_res_z_dr = dr::opaque<Float>((float)m_res_z);
        
        m_needs_sample_3 = true;
        
        // Check if medium is set
        if (m_medium) {
            Log(Info, "GridFluxMeter: grid [%s] to [%s], resolution (%d, %d, %d), %s, medium=%s",
                m_grid_min, m_grid_max, m_res_xy, m_res_xy, m_res_z,
                m_full_sphere ? "full sphere" : "hemisphere",
                m_medium->to_string().c_str());
        } else {
            Log(Warn, "GridFluxMeter: No medium specified! Rays won't interact with participating media. "
                "Add 'medium': {'type': 'ref', 'id': 'your_medium_id'} to the sensor.");
            Log(Info, "GridFluxMeter: grid [%s] to [%s], resolution (%d, %d, %d), %s",
                m_grid_min, m_grid_max, m_res_xy, m_res_xy, m_res_z,
                m_full_sphere ? "full sphere" : "hemisphere");
        }
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &position_sample,
                                          const Point2f &aperture_sample,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        auto [wavelengths, wav_weight] =
            sample_wavelengths(dr::zeros<SurfaceInteraction3f>(),
                               wavelength_sample, active);

        // position_sample ∈ [0,1]² maps to film pixel coordinates
        // Film pixel (px, py) where px ∈ [0, res_xy²), py ∈ [0, res_z)
        // px encodes (iy * res_xy + ix)
        
        Float px_f = position_sample.x() * m_film_width_dr;
        Float py_f = position_sample.y() * m_film_height_dr;
        
        // Get integer pixel indices
        Float px = dr::floor(px_f);
        Float py = dr::floor(py_f);
        
        // Clamp to valid range
        px = dr::clip(px, 0.f, m_film_width_dr - 1.f);
        py = dr::clip(py, 0.f, m_film_height_dr - 1.f);
        
        // Decompose px into (iy, ix) - px = iy * res_xy + ix
        Float iy = dr::floor(px / m_res_xy_dr);
        Float ix = px - iy * m_res_xy_dr;
        Float iz = py;
        
        // Grid cell centers: map integer index to [0,1] then to world
        Float tx = (ix + 0.5f) / m_res_xy_dr;
        Float ty = (iy + 0.5f) / m_res_xy_dr;
        Float tz = (iz + 0.5f) / m_res_z_dr;
        
        // World position
        Point3f origin(
            m_grid_min_x + tx * m_grid_size_x,
            m_grid_min_y + ty * m_grid_size_y,
            m_grid_min_z + tz * m_grid_size_z
        );
        
        // Sample direction
        Vector3f direction;
        if (m_full_sphere) {
            direction = warp::square_to_uniform_sphere(aperture_sample);
        } else {
            direction = warp::square_to_cosine_hemisphere(aperture_sample);
        }
        
        // Offset origin slightly in the ray direction
        Point3f ray_origin = origin + direction * math::RayEpsilon<Float>;
        
        Ray3f ray(ray_origin, direction, time, wavelengths);
        
        // Weight: 4π for full sphere, π for cosine hemisphere
        Float weight = m_full_sphere ? (4.f * dr::Pi<Float>) : dr::Pi<Float>;
        
        return { ray, depolarizer<Spectrum>(wav_weight * weight) };
    }

    std::pair<RayDifferential3f, Spectrum> sample_ray_differential(
        Float time, Float wavelength_sample,
        const Point2f &position_sample,
        const Point2f &aperture_sample,
        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        
        auto [ray, weight] = sample_ray(time, wavelength_sample, 
                                         position_sample, aperture_sample, active);
        
        RayDifferential3f ray_diff(ray);
        ray_diff.has_differentials = false;
        
        return { ray_diff, weight };
    }

    ScalarBoundingBox3f bbox() const override {
        return ScalarBoundingBox3f(m_grid_min, m_grid_max);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "GridFluxMeter[" << std::endl
            << "  grid_min = " << m_grid_min << "," << std::endl
            << "  grid_max = " << m_grid_max << "," << std::endl
            << "  resolution = (" << m_res_xy << ", " << m_res_xy << ", " << m_res_z << ")," << std::endl
            << "  full_sphere = " << (m_full_sphere ? "true" : "false") << "," << std::endl
            << "  medium = " << (m_medium ? m_medium->to_string() : "none") << "," << std::endl
            << "  film = " << m_film << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()

private:
    ScalarPoint3f m_grid_min;
    ScalarPoint3f m_grid_max;
    ScalarVector3f m_grid_size;
    int m_res_xy, m_res_z;
    int m_film_width, m_film_height;
    bool m_full_sphere;
    
    // Opaque drjit values to prevent kernel recompilation
    Float m_grid_min_x, m_grid_min_y, m_grid_min_z;
    Float m_grid_size_x, m_grid_size_y, m_grid_size_z;
    Float m_film_width_dr, m_film_height_dr;
    Float m_res_xy_dr, m_res_z_dr;
};

MI_EXPORT_PLUGIN(GridFluxMeter)
NAMESPACE_END(mitsuba)