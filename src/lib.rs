use geo::algorithm::triangulate_spade::{TriangulateSpade, SpadeTriangulationConfig};
use geo_types::{Coord, LineString, MultiPolygon, Polygon};
use ordered_float::OrderedFloat;
use rerun::Mesh3D;
use std::collections::HashMap;

type Vertex = [f64; 3];
type VertexKey = [OrderedFloat<f64>; 3];

pub trait ToMesh3D {
    fn to_mesh3d_on_sphere(
        &self,
        radius: f64,
        color: u32,
        max_subdivision_length: f64,
        subdivision_depth: u32,
    ) -> Mesh3D;
}

impl ToMesh3D for MultiPolygon<f64> {
    fn to_mesh3d_on_sphere(
        &self,
        radius: f64,
        color: u32,
        max_subdivision_length: f64,
        subdivision_depth: u32,
    ) -> Mesh3D {
        let mut vertices: Vec<VertexKey> = Vec::new();
        let mut normals: Vec<Vertex> = Vec::new();
        let mut indices: Vec<[u32; 3]> = Vec::new();
        let mut colors: Vec<u32> = Vec::new();
        let mut vertex_map: HashMap<VertexKey, u32> = HashMap::new();
        let mut vertex_index: u32 = 0;

        // Iterate over polygons in MultiPolygon
        for polygon in &self.0 {
            // Subdivide the polygon's rings
            let subdivided_exterior =
                subdivide_line_string(&polygon.exterior(), max_subdivision_length);
            let subdivided_interiors = polygon
                .interiors()
                .iter()
                .map(|ring| subdivide_line_string(ring, max_subdivision_length))
                .collect::<Vec<_>>();

            let subdivided_polygon = Polygon::new(subdivided_exterior, subdivided_interiors);

            let config = SpadeTriangulationConfig::default();
            let triangles = match subdivided_polygon.constrained_triangulation(config) {
                Ok(triangles) => triangles,
                Err(e) => {
                    eprintln!("Warning: Failed to triangulate polygon: {:?}", e);
                    continue;
                }
            };

            for triangle in triangles {
                let coords = [triangle.0, triangle.1, triangle.2];
                subdivide_triangle_on_sphere(
                    coords,
                    subdivision_depth,
                    radius,
                    &mut vertices,
                    &mut normals,
                    &mut indices,
                    &mut colors,
                    &mut vertex_map,
                    &mut vertex_index,
                    color,
                );
            }
        }

        let vertices_f32: Vec<[f32; 3]> = vertices
            .iter()
            .map(|&[x, y, z]| [x.0 as f32, y.0 as f32, z.0 as f32])
            .collect();

        let normals_f32: Vec<[f32; 3]> = normals
            .iter()
            .map(|&[x, y, z]| [x as f32, y as f32, z as f32])
            .collect();

        Mesh3D::new(vertices_f32)
            .with_vertex_normals(normals_f32)
            .with_vertex_colors(colors)
            .with_triangle_indices(indices)
    }
}

// Function to subdivide LineString (edge subdivision remains the same)
fn subdivide_line_string(
    line_string: &LineString<f64>,
    max_distance: f64,
) -> LineString<f64> {
    let mut new_coords = Vec::new();
    for window in line_string.0.windows(2) {
        let start = window[0];
        let end = window[1];
        new_coords.push(start);

        let segment_length = haversine_distance(start, end);
        if segment_length > max_distance {
            let num_subdivisions = (segment_length / max_distance).ceil() as usize;
            for i in 1..num_subdivisions {
                let t = i as f64 / num_subdivisions as f64;
                let interpolated = interpolate_coord(start, end, t);
                new_coords.push(interpolated);
            }
        }
    }
    // Add the last point
    if let Some(last) = line_string.0.last() {
        new_coords.push(*last);
    }
    LineString(new_coords)
}

// Haversine distance between two coordinates (in meters)
fn haversine_distance(c1: Coord<f64>, c2: Coord<f64>) -> f64 {
    let lat1 = c1.y.to_radians();
    let lon1 = c1.x.to_radians();
    let lat2 = c2.y.to_radians();
    let lon2 = c2.x.to_radians();

    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let a = (dlat / 2.0).sin().powi(2)
        + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    // Earth's radius in meters
    let r = 6_371_000.0;
    r * c
}

// Linear interpolation between two coordinates
fn interpolate_coord(c1: Coord<f64>, c2: Coord<f64>, t: f64) -> Coord<f64> {
    Coord {
        x: c1.x + (c2.x - c1.x) * t,
        y: c1.y + (c2.y - c1.y) * t,
    }
}

// Function to subdivide a triangle on the sphere
fn subdivide_triangle_on_sphere(
    coords: [Coord<f64>; 3],
    depth: u32,
    radius: f64,
    vertices: &mut Vec<[OrderedFloat<f64>; 3]>,
    normals: &mut Vec<[f64; 3]>,
    indices: &mut Vec<[u32; 3]>,
    colors: &mut Vec<u32>,
    vertex_map: &mut HashMap<[OrderedFloat<f64>; 3], u32>,
    vertex_index: &mut u32,
    color: u32,
) {
    if depth == 0 {
        // Base case: map the triangle to the sphere and add it to the mesh
        let idxs = coords
            .iter()
            .map(|coord| {
                // Normalize longitude to [-180, 180]
                let lon_normalized = if coord.x > 180.0 {
                    coord.x - 360.0
                } else if coord.x < -180.0 {
                    coord.x + 360.0
                } else {
                    coord.x
                };

                // Convert lat/lon to 3D coordinates on the sphere
                let lat_rad = coord.y.to_radians();
                let lon_rad = lon_normalized.to_radians();

                let x = lat_rad.cos() * lon_rad.cos();
                let y = lat_rad.cos() * lon_rad.sin();
                let z = lat_rad.sin();

                let vertex = [x, y, z];

                // Use ordered float
                let vertex = [
                    OrderedFloat(vertex[0]),
                    OrderedFloat(vertex[1]),
                    OrderedFloat(vertex[2]),
                ];

                // Ensure uniqueness of vertices
                if let Some(&idx) = vertex_map.get(&vertex) {
                    idx
                } else {
                    let idx = *vertex_index;
                    *vertex_index += 1;

                    // Scale vertex to sphere radius
                    let vertex_scaled = [vertex[0] * radius, vertex[1] * radius, vertex[2] * radius];
                    vertices.push(vertex_scaled);

                    // Normal is the normalized vertex
                    normals.push([vertex[0].0, vertex[1].0, vertex[2].0]);

                    colors.push(color);

                    vertex_map.insert(vertex, idx);

                    idx
                }
            })
            .collect::<Vec<u32>>();

        indices.push([idxs[0], idxs[1], idxs[2]]);
    } else {
        // Recursive subdivision
        let mid01 = mid_coord(coords[0], coords[1]);
        let mid12 = mid_coord(coords[1], coords[2]);
        let mid20 = mid_coord(coords[2], coords[0]);

        subdivide_triangle_on_sphere(
            [coords[0], mid01, mid20],
            depth - 1,
            radius,
            vertices,
            normals,
            indices,
            colors,
            vertex_map,
            vertex_index,
            color,
        );
        subdivide_triangle_on_sphere(
            [coords[1], mid12, mid01],
            depth - 1,
            radius,
            vertices,
            normals,
            indices,
            colors,
            vertex_map,
            vertex_index,
            color,
        );
        subdivide_triangle_on_sphere(
            [coords[2], mid20, mid12],
            depth - 1,
            radius,
            vertices,
            normals,
            indices,
            colors,
            vertex_map,
            vertex_index,
            color,
        );
        subdivide_triangle_on_sphere(
            [mid01, mid12, mid20],
            depth - 1,
            radius,
            vertices,
            normals,
            indices,
            colors,
            vertex_map,
            vertex_index,
            color,
        );
    }
}

// Function to compute the midpoint between two coordinates on the sphere
fn mid_coord(c1: Coord<f64>, c2: Coord<f64>) -> Coord<f64> {
    // Convert to Cartesian coordinates
    let lat1 = c1.y.to_radians();
    let lon1 = c1.x.to_radians();
    let x1 = lat1.cos() * lon1.cos();
    let y1 = lat1.cos() * lon1.sin();
    let z1 = lat1.sin();

    let lat2 = c2.y.to_radians();
    let lon2 = c2.x.to_radians();
    let x2 = lat2.cos() * lon2.cos();
    let y2 = lat2.cos() * lon2.sin();
    let z2 = lat2.sin();

    // Compute midpoint in Cartesian coordinates
    let x_mid = (x1 + x2) / 2.0;
    let y_mid = (y1 + y2) / 2.0;
    let z_mid = (z1 + z2) / 2.0;

    // Normalize to lie on the unit sphere
    let mag = (x_mid.powi(2) + y_mid.powi(2) + z_mid.powi(2)).sqrt();
    let x_mid_norm = x_mid / mag;
    let y_mid_norm = y_mid / mag;
    let z_mid_norm = z_mid / mag;

    // Convert back to latitude and longitude
    let lat_mid = z_mid_norm.asin().to_degrees();
    let lon_mid = y_mid_norm.atan2(x_mid_norm).to_degrees();

    Coord { x: lon_mid, y: lat_mid }
}

pub fn plot_shapefile(
    rec: &rerun::RecordingStream,
    name: &str,
    shapefile_path: &str,
    color: u32,
    sphere_radius: f64,
    max_subdivision_length: f64,
    subdivision_depth: u32,
) {
    let mut reader = shapefile::Reader::from_path(shapefile_path)
        .expect("Error reading shapefile");
    for (i, shape_record) in reader.iter_shapes_and_records().enumerate() {
        let (shape, _record) = shape_record.expect("Error reading shape record");
        let geometry = geo_types::Geometry::<f64>::try_from(shape).unwrap();
        let multi_polygon: MultiPolygon<f64> = match geometry {
            geo_types::Geometry::MultiPolygon(multi_polygon) => multi_polygon,
            geo_types::Geometry::Polygon(polygon) => MultiPolygon(vec![polygon]),
            _ => panic!("Unexpected geometry type"),
        };

        let mesh = multi_polygon.to_mesh3d_on_sphere(
            sphere_radius,
            color,
            max_subdivision_length,
            subdivision_depth,
        );
        rec.log_static(format!("{name}/{i}"), &mesh)
            .expect("Error logging mesh");
    }
}



/// Converts latitude and longitude in degrees to 3D Cartesian coordinates on a sphere.
///
/// # Arguments
///
/// * `lat_deg` - Latitude in degrees.
/// * `lon_deg` - Longitude in degrees.
/// * `radius` - Radius of the sphere.
///
/// # Returns
///
/// A `[f64; 3]` array representing the (x, y, z) coordinates.
pub fn lat_lon_to_xyz(lat_deg: f64, lon_deg: f64, radius: f64) -> [f64; 3] {
    let lat_rad = lat_deg.to_radians();
    let lon_rad = lon_deg.to_radians();

    let x = radius * lat_rad.cos() * lon_rad.cos();
    let y = radius * lat_rad.cos() * lon_rad.sin();
    let z = radius * lat_rad.sin();

    [x, y, z]
}
