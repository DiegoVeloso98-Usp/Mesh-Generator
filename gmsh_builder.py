import gmsh
import sys
import json
import os
import math

# ==============================================================================
# GMSH MESHING SCRIPT - PHASE 3: INCLUSIONS
# ==============================================================================
# This script builds upon the previous version by adding inclusions.
# Key changes:
# 1. New "inclusions" section in the JSON for defining polygons and ellipses.
# 2. Inclusions can be defined as holes (`is_hole: true`) or new subdomains.
# 3. Uses `gmsh.model.occ.cut` for holes and `fragment` for new subdomains.
# 4. Logic to automatically find the parent subdomain for each inclusion.
# 5. FIX: Correctly creates a curve loop from an ellipse before creating a surface.
# ==============================================================================


class GmshGenerator:
    """
    A class to handle the creation of a Gmsh model from a JSON input.
    It encapsulates the entire process from geometry creation to meshing.
    """
    def __init__(self, input_data):
        """Initializes the generator with input data and sets up Gmsh."""
        self.data = input_data
        self.point_map = {}
        self.curve_map = {}
        # This will store the final (dim, tag) of surfaces for physical group assignment
        self.final_surfaces = {} 
        
        gmsh.initialize()
        gmsh.model.add("main_model")

    def _create_points(self):
        """Creates Gmsh points using the OCC kernel."""
        print("Creating points...")
        point_definitions = self.data["points"]
        for tag, coords in point_definitions.items():
            x, y, z, mesh_size = coords
            point_id = gmsh.model.occ.addPoint(x, y, z, mesh_size)
            self.point_map[tag] = point_id

    def _create_curves(self):
        """Creates Gmsh curves (lines/arcs) using the OCC kernel."""
        print("Creating curves...")
        curve_definitions = self.data["curves"]
        for tag, definition in curve_definitions.items():
            curve_type = definition["type"]
            point_tags = definition["points"]
            
            gmsh_point_ids = [self.point_map[pt] for pt in point_tags]

            if curve_type == "line":
                curve_id = gmsh.model.occ.addLine(gmsh_point_ids[0], gmsh_point_ids[1])
            elif curve_type == "arc":
                curve_id = gmsh.model.occ.addCircleArc(gmsh_point_ids[0], gmsh_point_ids[1], gmsh_point_ids[2])
            else:
                raise ValueError(f"Unknown curve type: {curve_type}")
            
            self.curve_map[tag] = curve_id
        
        gmsh.model.occ.synchronize()

    def _create_inclusions(self):
        """Creates geometry for all defined inclusions (polygons, ellipses)."""
        if "inclusions" not in self.data:
            return {}

        print("Creating inclusions...")
        inclusion_map = {}
        for name, props in self.data["inclusions"].items():
            if props["type"] == "polygon":
                points = [gmsh.model.occ.addPoint(p[0], p[1], p[2], props["mesh_size"]) for p in props["points"]]
                lines = [gmsh.model.occ.addLine(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
                loop = gmsh.model.occ.addCurveLoop(lines)
                surface = gmsh.model.occ.addPlaneSurface([loop])
                inclusion_map[name] = (2, surface)

            elif props["type"] == "ellipse":
                center = props["center"]
                rx = props["radius_x"]
                ry = props["radius_y"]
                angle_deg = props.get("rotation_angle", 0)

                # Create the ellipse curve
                ellipse_curve = gmsh.model.occ.addEllipse(center[0], center[1], center[2], rx, ry)
                # FIX: An ellipse curve must be put into a curve loop before making a surface
                ellipse_loop = gmsh.model.occ.addCurveLoop([ellipse_curve])
                surface = gmsh.model.occ.addPlaneSurface([ellipse_loop])
                
                if angle_deg != 0:
                    gmsh.model.occ.rotate([(2, surface)], center[0], center[1], center[2], 0, 0, 1, math.radians(angle_deg))
                
                inclusion_map[name] = (2, surface)
        
        gmsh.model.occ.synchronize()
        return inclusion_map

    def _create_and_process_geometry(self):
        """The main geometry creation and processing function."""
        print("Creating base domains...")
        
        # 1. Create the outer boundary and fragment it into subdomains
        subdomain_definitions = self.data["subdomains"]
        all_subdomain_curves = set(tag for info in subdomain_definitions.values() for tag in info["curves"])

        curve_counts = {tag: sum(1 for info in subdomain_definitions.values() if tag in info["curves"]) for tag in all_subdomain_curves}
        
        exterior_curve_tags = [tag for tag, count in curve_counts.items() if count == 1]
        internal_curve_tags = [tag for tag, count in curve_counts.items() if count > 1]

        exterior_gmsh_ids = [self.curve_map[tag] for tag in exterior_curve_tags]
        internal_gmsh_ids = [self.curve_map[tag] for tag in internal_curve_tags]

        # Sort exterior curves to form a closed loop
        sorted_exterior_curves = self._sort_curves_into_loop(exterior_gmsh_ids)
        
        curve_loop = gmsh.model.occ.addCurveLoop(sorted_exterior_curves)
        main_surface = gmsh.model.occ.addPlaneSurface([curve_loop])
        gmsh.model.occ.synchronize()

        # Fragment the main surface by internal lines
        tool_dim_tags = [(1, curve_id) for curve_id in internal_gmsh_ids]
        fragmented_entities, _ = gmsh.model.occ.fragment([(2, main_surface)], tool_dim_tags)
        gmsh.model.occ.synchronize()

        # Store the initial subdomains for later processing
        current_surfaces = [entity[1] for entity in fragmented_entities if entity[0] == 2]

        # 2. Create inclusion geometries
        inclusion_map = self._create_inclusions()

        # 3. Process inclusions (cut or fragment)
        if not inclusion_map:
            # If no inclusions, the current surfaces are the final ones
            self.final_surfaces = gmsh.model.getEntities(dim=2)
            return

        print("Processing inclusions...")
        objects_to_fragment = list(current_surfaces)
        tools_to_fragment_with = []

        for name, (dim, tag) in inclusion_map.items():
            props = self.data["inclusions"][name]
            if props["is_hole"]:
                # Cut the hole from all potential parent surfaces
                objects_to_fragment, _ = gmsh.model.occ.cut([(2, s) for s in objects_to_fragment], [(dim, tag)])
                objects_to_fragment = [s[1] for s in objects_to_fragment] # Get just the tags
            else:
                # Add non-hole inclusions to the list of tools for final fragmentation
                tools_to_fragment_with.append((dim, tag))

        # Final fragmentation step if there were non-hole inclusions
        if tools_to_fragment_with:
            final_entities, _ = gmsh.model.occ.fragment([(2, s) for s in objects_to_fragment], tools_to_fragment_with)
            self.final_surfaces = [e for e in final_entities if e[0] == 2]
        else:
            self.final_surfaces = [(2, s) for s in objects_to_fragment]

        gmsh.model.occ.synchronize()


    def _sort_curves_into_loop(self, curve_ids):
        """Helper function to sort a list of curve IDs into a continuous loop."""
        if not curve_ids:
            return []
        
        curve_endpoints = {}
        for cid in curve_ids:
            boundary_vtx = gmsh.model.getBoundary([(1, cid)], combined=False, oriented=True)
            start_vtx_tag = boundary_vtx[0][1]
            end_vtx_tag = boundary_vtx[1][1]
            curve_endpoints[cid] = (start_vtx_tag, end_vtx_tag)
        
        sorted_curves = [curve_ids[0]]
        unsorted_curves = curve_ids[1:]
        
        while unsorted_curves:
            last_curve_end_vtx = curve_endpoints[sorted_curves[-1]][1]
            found_next = False
            for i, next_curve_id in enumerate(unsorted_curves):
                next_start_vtx, _ = curve_endpoints[next_curve_id]
                if next_start_vtx == last_curve_end_vtx:
                    sorted_curves.append(unsorted_curves.pop(i))
                    found_next = True
                    break
            if not found_next:
                raise Exception("Could not form a closed loop from exterior curves.")
        return sorted_curves
    def _assign_physical_groups(self):
        """Assigns physical tags to the final surfaces and boundary lines."""
        print("Assigning physical groups...")
        
        tagged_surfaces = set()

        # Debug: Print all surfaces and their properties
        print(f"Total surfaces to tag: {len(self.final_surfaces)}")
        for i, (surface_dim, surface_tag) in enumerate(self.final_surfaces):
            com = gmsh.model.occ.getCenterOfMass(surface_dim, surface_tag)
            area = gmsh.model.occ.getMass(surface_dim, surface_tag)
            print(f"Surface {surface_tag}: COM=({com[0]:.3f}, {com[1]:.3f}), Area={area:.4f}")

        # 1. Process non-hole inclusions first - these should be the smallest, most precisely defined
        inclusions = {name: props for name, props in self.data.get("inclusions", {}).items() if not props.get("is_hole", False)}
        
        for name, domain_info in inclusions.items():
            physical_tag = domain_info["tag"]
            best_match = None
            best_score = float('inf')
            
            for surface_dim, surface_tag in self.final_surfaces:
                if surface_tag in tagged_surfaces:
                    continue
                
                com = gmsh.model.occ.getCenterOfMass(surface_dim, surface_tag)
                area = gmsh.model.occ.getMass(surface_dim, surface_tag)
                
                if domain_info["type"] == "polygon":
                    polygon_points = domain_info["points"]
                    expected_area = self._polygon_area(polygon_points)
                    
                    # Calculate polygon centroid
                    poly_centroid_x = sum(p[0] for p in polygon_points) / len(polygon_points)
                    poly_centroid_y = sum(p[1] for p in polygon_points) / len(polygon_points)
                    
                    # Distance from surface COM to polygon centroid
                    dist = math.sqrt((com[0] - poly_centroid_x)**2 + (com[1] - poly_centroid_y)**2)
                    area_diff = abs(area - expected_area) / expected_area
                    
                    # Score based on distance and area difference
                    score = dist + area_diff
                    
                    if score < best_score and area_diff < 0.2:  # 20% area tolerance
                        best_score = score
                        best_match = surface_tag
                
                elif domain_info["type"] == "ellipse":
                    center = domain_info["center"]
                    rx, ry = domain_info["radius_x"], domain_info["radius_y"]
                    expected_area = math.pi * rx * ry
                    
                    # Distance from surface COM to ellipse center
                    dist = math.sqrt((com[0] - center[0])**2 + (com[1] - center[1])**2)
                    area_diff = abs(area - expected_area) / expected_area
                    
                    # Score based on distance and area difference
                    score = dist + area_diff
                    
                    if score < best_score and area_diff < 0.2:  # 20% area tolerance
                        best_score = score
                        best_match = surface_tag
            
            if best_match is not None:
                gmsh.model.addPhysicalGroup(2, [best_match], physical_tag)
                gmsh.model.setPhysicalName(2, physical_tag, name)
                tagged_surfaces.add(best_match)
                com = gmsh.model.occ.getCenterOfMass(2, best_match)
                area = gmsh.model.occ.getMass(2, best_match)
                print(f"Tagged surface {best_match} as {name} (COM: {com[0]:.3f}, {com[1]:.3f}, area: {area:.4f})")

        # 2. Process main subdomains - identify by geometric position
        # For this specific case, we know LeftDomain should be on the left (x < 1) and RightDomain on the right (x > 1)
        subdomains = self.data["subdomains"]
        
        for name, domain_info in subdomains.items():
            physical_tag = domain_info["tag"]
            
            for surface_dim, surface_tag in self.final_surfaces:
                if surface_tag in tagged_surfaces:
                    continue
                
                com = gmsh.model.occ.getCenterOfMass(surface_dim, surface_tag)
                area = gmsh.model.occ.getMass(surface_dim, surface_tag)
                
                # Use geometric positioning to identify domains
                is_match = False
                
                if name == "LeftDomain":
                    # LeftDomain should have COM with x < 1.0 and reasonable area
                    if com[0] < 1.0 and area > 0.2:  # Reasonable area threshold
                        is_match = True
                
                elif name == "RightDomain":
                    # RightDomain should have COM with x > 1.0 and reasonable area
                    if com[0] > 1.0 and area > 0.2:  # Reasonable area threshold
                        is_match = True
                
                if is_match:
                    gmsh.model.addPhysicalGroup(2, [surface_tag], physical_tag)
                    gmsh.model.setPhysicalName(2, physical_tag, name)
                    tagged_surfaces.add(surface_tag)
                    print(f"Tagged surface {surface_tag} as {name} (COM: {com[0]:.3f}, {com[1]:.3f}, area: {area:.4f})")
                    break

        # 3. Tag any remaining untagged surfaces with a default tag for debugging
        for surface_dim, surface_tag in self.final_surfaces:
            if surface_tag not in tagged_surfaces:
                com = gmsh.model.occ.getCenterOfMass(surface_dim, surface_tag)
                area = gmsh.model.occ.getMass(surface_dim, surface_tag)
                print(f"WARNING: Untagged surface {surface_tag} (COM: {com[0]:.3f}, {com[1]:.3f}, area: {area:.4f})")

        # 4. Process boundary conditions - use original curve mapping
        for bc_name, bc_info in self.data["boundary_conditions"].items():
            bc_curve_ids = [self.curve_map[tag] for tag in bc_info["curves"]]
            bc_tag = bc_info["tag"]
            gmsh.model.addPhysicalGroup(1, bc_curve_ids, bc_tag)
            gmsh.model.setPhysicalName(1, bc_tag, bc_name)



    def _polygon_area(self, polygon_points):
        """Calculate the area of a polygon using the shoelace formula."""
        n = len(polygon_points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon_points[i][0] * polygon_points[j][1]
            area -= polygon_points[j][0] * polygon_points[i][1]
        return abs(area) / 2.0

    def generate_mesh(self):
        """Runs the entire mesh generation pipeline."""
        self._create_points()
        self._create_curves()
        self._create_and_process_geometry()
        self._assign_physical_groups()
        
        print("Generating 2D mesh...")
        mesh_opts = self.data["mesh_options"]
        gmsh.option.setNumber("Mesh.Algorithm", 5)
        if mesh_opts.get("element_type") == "Q4":
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 8)


        gmsh.model.mesh.generate(2)
        print("Mesh generation complete.")

        if mesh_opts.get("show_gui", False) and '-nopopup' not in sys.argv:
            gmsh.fltk.run()

    def close(self):
        """Finalizes the Gmsh instance."""
        gmsh.finalize()


def main(input_file_path):
    """Main execution function."""
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at '{input_file_path}'")
        return

    with open(input_file_path, 'r') as f:
        mesh_input = json.load(f)

    generator = GmshGenerator(mesh_input)
    try:
        generator.generate_mesh()
    finally:
        generator.close()



if __name__ == "__main__":
    # The script expects 'input.json' to be in the same directory.
    json_path = "other.json"
    main(json_path)
