// cgal_polygonal_reconstruction.cpp
// CGAL Polygonal Surface Reconstruction CLI wrapper (auto plane detection via region growing).
// Derived from CGAL 5.4 example polyfit_example_with_region_growing.cpp (GPL-3+/LGPL-3+).
//
// Usage:
//   cgal_polygonal_reconstruction <input.ply> <output.obj|off|ply>
//       [-d max_distance_to_plane=0.02] [-a max_accepted_angle_deg=25]
//       [-m min_region_size=100] [-r search_sphere_radius=0.03]

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/read_points.h>
#include <CGAL/property_map.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing_on_point_set.h>
#include <CGAL/Polygonal_surface_reconstruction.h>

#ifdef CGAL_USE_SCIP
#include <CGAL/SCIP_mixed_integer_program_traits.h>
typedef CGAL::SCIP_mixed_integer_program_traits<double>     MIP_Solver;
#elif defined(CGAL_USE_GLPK)
#include <CGAL/GLPK_mixed_integer_program_traits.h>
typedef CGAL::GLPK_mixed_integer_program_traits<double>     MIP_Solver;
#else
#error "Either CGAL_USE_SCIP or CGAL_USE_GLPK must be defined"
#endif

#include <CGAL/Timer.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::FT                                          FT;
typedef Kernel::Point_3                                     Point;
typedef Kernel::Vector_3                                    Vector;

typedef boost::tuple<Point, Vector, int>                    PNI;
typedef std::vector<PNI>                                    Point_vector;
typedef CGAL::Nth_of_tuple_property_map<0, PNI>             Point_map;
typedef CGAL::Nth_of_tuple_property_map<1, PNI>             Normal_map;
typedef CGAL::Nth_of_tuple_property_map<2, PNI>             Plane_index_map;

typedef CGAL::Shape_detection::Point_set::
    Sphere_neighbor_query<Kernel, Point_vector, Point_map>  Neighbor_query;
typedef CGAL::Shape_detection::Point_set::
    Least_squares_plane_fit_region<Kernel, Point_vector, Point_map, Normal_map> Region_type;
typedef CGAL::Shape_detection::
    Region_growing<Point_vector, Neighbor_query, Region_type> Region_growing;

typedef CGAL::Surface_mesh<Point>                           Surface_mesh;
typedef CGAL::Polygonal_surface_reconstruction<Kernel>      Polygonal_surface_reconstruction;

class Index_map {
public:
  using key_type = std::size_t;
  using value_type = int;
  using reference = value_type;
  using category = boost::readable_property_map_tag;

  Index_map() { }
  template<typename PointRange>
  Index_map(const PointRange& points,
            const std::vector< std::vector<std::size_t> >& regions)
    : m_indices(new std::vector<int>(points.size(), -1))
  {
    for (std::size_t i = 0; i < regions.size(); ++i)
      for (const std::size_t idx : regions[i])
        (*m_indices)[idx] = static_cast<int>(i);
  }

  inline friend value_type get(const Index_map& index_map, const key_type key)
  { return (*(index_map.m_indices))[key]; }

private:
  std::shared_ptr< std::vector<int> > m_indices;
};

static void write_obj(const Surface_mesh& mesh, const std::string& path) {
  std::ofstream out(path.c_str());
  if (!out) { std::cerr << "Cannot open output file: " << path << std::endl; std::exit(EXIT_FAILURE); }
  out << "# CGAL Polygonal Surface Reconstruction\n";
  std::vector<std::size_t> reindex(mesh.num_vertices(), 0);
  std::size_t next = 1;
  for (const auto& v : mesh.vertices()) {
    const Point& p = mesh.point(v);
    out << "v " << p.x() << " " << p.y() << " " << p.z() << "\n";
    reindex[v] = next++;
  }
  // Fan-triangulate: Open3D skips non-triangle primitives on OBJ load.
  for (const auto& f : mesh.faces()) {
    std::vector<std::size_t> loop;
    for (const auto& vtx : CGAL::vertices_around_face(mesh.halfedge(f), mesh))
      loop.push_back(reindex[vtx]);
    if (loop.size() < 3) continue;
    for (std::size_t i = 1; i + 1 < loop.size(); ++i)
      out << "f " << loop[0] << " " << loop[i] << " " << loop[i + 1] << "\n";
  }
}

int main(int argc, char** argv)
{
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <input.ply> <output.obj|off|ply>"
              << " [-d max_distance_to_plane] [-a max_accepted_angle_deg] [-m min_region_size]"
              << " [-r search_sphere_radius]" << std::endl;
    return EXIT_FAILURE;
  }
  const std::string input_file  = argv[1];
  const std::string output_file = argv[2];

  FT max_distance_to_plane = FT(2) / FT(100);   // 0.02 m
  FT max_accepted_angle    = FT(25);
  std::size_t min_region_size = 100;
  FT search_sphere_radius  = FT(3) / FT(100);   // 0.03 m

  for (int i = 3; i < argc - 1; i += 2) {
    const std::string key = argv[i];
    const double val = std::atof(argv[i + 1]);
    if      (key == "-d") max_distance_to_plane = val;
    else if (key == "-a") max_accepted_angle = val;
    else if (key == "-m") min_region_size = static_cast<std::size_t>(val);
    else if (key == "-r") search_sphere_radius = val;
    else { std::cerr << "Unknown option: " << key << std::endl; return EXIT_FAILURE; }
  }

  CGAL::Timer t;
  Point_vector points;
  t.start();
  if (!CGAL::IO::read_points(input_file.c_str(), std::back_inserter(points),
                             CGAL::parameters::point_map(Point_map()).normal_map(Normal_map()))) {
    std::cerr << "Error: cannot read file " << input_file << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Loaded " << points.size() << " points (" << t.time() << " sec)" << std::endl;
  if (points.empty()) return EXIT_FAILURE;

  // Shape detection (plane extraction via region growing)
  Neighbor_query neighbor_query(points, search_sphere_radius);
  Region_type region_type(points, max_distance_to_plane, max_accepted_angle, min_region_size);
  Region_growing region_growing(points, neighbor_query, region_type);

  std::cout << "Extracting planes..." << std::endl;
  t.reset();
  std::vector< std::vector<std::size_t> > regions;
  region_growing.detect(std::back_inserter(regions));
  std::cout << regions.size() << " planes extracted (" << t.time() << " sec)" << std::endl;
  if (regions.empty()) {
    std::cerr << "No planes detected." << std::endl;
    return EXIT_FAILURE;
  }

  Index_map index_map(points, regions);
  for (std::size_t i = 0; i < points.size(); ++i)
    points[i].get<2>() = get(index_map, i);

  // Reconstruction
  std::cout << "Generating candidate faces & solving MIP..." << std::endl;
  t.reset();
  Polygonal_surface_reconstruction algo(points, Point_map(), Normal_map(), Plane_index_map());
  Surface_mesh model;
  if (!algo.reconstruct<MIP_Solver>(model)) {
    std::cerr << "Reconstruction failed: " << algo.error_message() << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << model.number_of_faces() << " faces (" << t.time() << " sec)" << std::endl;

  // Save (extension-driven; OBJ written manually as triangle/polygon soup)
  if (output_file.size() >= 4 && output_file.compare(output_file.size() - 4, 4, ".obj") == 0) {
    write_obj(model, output_file);
    std::cout << "Saved to " << output_file << std::endl;
  } else if (output_file.size() >= 4 && output_file.compare(output_file.size() - 4, 4, ".ply") == 0) {
    if (!CGAL::IO::write_PLY(output_file, model)) {
      std::cerr << "Failed saving " << output_file << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "Saved to " << output_file << std::endl;
  } else {
    if (!CGAL::IO::write_OFF(output_file, model)) {
      std::cerr << "Failed saving " << output_file << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "Saved to " << output_file << std::endl;
  }
  return EXIT_SUCCESS;
}
