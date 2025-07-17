"""
Stage 06: Constraint Enforcement
Validates multi-tile coordination results and enforces quality constraints.
Critical quality gate for the multi-tile diffusion system.
"""

import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple
from skimage.metrics import structural_similarity as ssim
import cv2
import structlog
from scipy.spatial.distance import euclidean
from colorspacious import cspace_convert

from ..core.pipeline_context import PipelineContext
from ..core.model_registry import ModelRegistry

logger = structlog.get_logger()

def execute(context: PipelineContext) -> Dict[str, Any]:
    """
    Execute Stage 06: Constraint Enforcement
    
    Validates the results of multi-tile coordination from Stage 05:
    - Shared edge matching from round-robin edge copying
    - Cross-tile attention window coherence
    - Palette compliance across simultaneous generation
    - Structural compliance with ControlNet guidance
    - Global consistency validation
    
    Args:
        context: Pipeline context with generated tiles and coordination data
        
    Returns:
        Dict with comprehensive constraint validation results
    """
    logger.info("Starting constraint enforcement", job_id=context.get_job_id())
    
    try:
        # Validate context state
        if not context.validate_context_state(6):
            return {"success": False, "errors": context.pipeline_errors}
        
        # Get required data
        generated_tiles = context.generated_tiles
        tileset_setup = context.universal_tileset
        extracted_config = getattr(context, 'extracted_config', None)
        generation_metadata = context.generation_metadata
        
        if not generated_tiles:
            context.add_error(6, "No generated tiles found")
            return {"success": False, "errors": ["No generated tiles found"]}
        
        # Initialize comprehensive constraint enforcer
        enforcer = ComprehensiveConstraintEnforcer(
            tileset_setup=tileset_setup,
            config=extracted_config,
            generation_metadata=generation_metadata
        )
        
        # Run comprehensive constraint validation
        validation_result = enforcer.validate_multi_tile_coordination(generated_tiles)
        
        # Store results in context
        context.constraint_violations = validation_result["violations"]
        context.constraint_scores = validation_result["scores"]
        
        # Mark tiles for regeneration based on validation results
        regeneration_decisions = enforcer.make_regeneration_decisions(validation_result)
        
        for tile_id, should_regenerate in regeneration_decisions.items():
            if should_regenerate["regenerate"]:
                context.mark_tile_for_regeneration(tile_id, should_regenerate["reason"])
        
        logger.info("Constraint enforcement completed", 
                   job_id=context.get_job_id(),
                   tiles_validated=len(generated_tiles),
                   edge_violations=validation_result["summary"]["edge_violations"],
                   coordination_score=validation_result["summary"]["coordination_effectiveness"],
                   tiles_marked_for_regen=len(context.regeneration_queue))
        
        return {
            "success": True,
            "constraint_violations": validation_result["violations"],
            "constraint_scores": validation_result["scores"],
            "coordination_validation": validation_result["coordination_validation"],
            "validation_summary": validation_result["summary"],
            "regeneration_decisions": regeneration_decisions,
            "tiles_marked_for_regeneration": context.regeneration_queue
        }
        
    except Exception as e:
        error_msg = f"Constraint enforcement failed: {str(e)}"
        logger.error("Constraint enforcement failed", job_id=context.get_job_id(), error=error_msg)
        context.add_error(6, error_msg)
        return {"success": False, "errors": [error_msg]}

class ComprehensiveConstraintEnforcer:
    """Comprehensive constraint enforcer for multi-tile coordination validation."""
    
    def __init__(self, tileset_setup: Any, config: Dict[str, Any], generation_metadata: Dict[str, Any]):
        self.tileset_setup = tileset_setup
        self.config = config
        self.generation_metadata = generation_metadata
        
        # Get constraint thresholds
        constraints = config.get("constraints", {})
        self.edge_similarity_threshold = constraints.get("edge_similarity", 0.98)
        self.palette_deviation_threshold = constraints.get("palette_deviation", 3.0)
        self.structural_compliance_threshold = constraints.get("structural_compliance", 1.0)
        self.sub_tile_coherence_threshold = constraints.get("sub_tile_coherence", 0.7)
        
        # Multi-tile coordination thresholds
        self.attention_window_coherence_threshold = 0.85
        self.cross_tile_consistency_threshold = 0.8
        
        # Get coordination data
        self.adjacency_graph = tileset_setup.get_adjacency_graph()
        self.shared_edges = tileset_setup.get_shared_edges()
        self.atlas_layout = tileset_setup.get_atlas_layout()
        
        # Model registry for palette validation
        self.model_registry = ModelRegistry()
    
    def validate_multi_tile_coordination(self, tiles: Dict[int, Image.Image]) -> Dict[str, Any]:
        """Comprehensive validation of multi-tile coordination results."""
        violations = {}
        scores = {}
        coordination_validation = {}
        
        # 1. CRITICAL: Validate shared edge matching (should be near-perfect with round-robin copying)
        edge_validation = self._validate_shared_edge_matching(tiles)
        coordination_validation["shared_edge_matching"] = edge_validation
        
        # 2. Validate cross-tile attention window coherence
        attention_validation = self._validate_attention_window_coherence(tiles)
        coordination_validation["attention_window_coherence"] = attention_validation
        
        # 3. Validate global consistency from simultaneous generation
        consistency_validation = self._validate_global_consistency(tiles)
        coordination_validation["global_consistency"] = consistency_validation
        
        # 4. Individual tile validation
        for tile_id, tile_image in tiles.items():
            tile_violations = []
            tile_scores = {}
            
            # Palette compliance
            palette_result = self._validate_palette_compliance(tile_id, tile_image)
            if not palette_result["passed"]:
                tile_violations.extend(palette_result["violations"])
            tile_scores.update(palette_result["scores"])
            
            # Structural compliance with ControlNet guidance
            structural_result = self._validate_structural_compliance(tile_id, tile_image)
            if not structural_result["passed"]:
                tile_violations.extend(structural_result["violations"])
            tile_scores.update(structural_result["scores"])
            
            # Sub-tile coherence
            coherence_result = self._validate_sub_tile_coherence(tile_id, tile_image)
            if not coherence_result["passed"]:
                tile_violations.extend(coherence_result["violations"])
            tile_scores.update(coherence_result["scores"])
            
            violations[tile_id] = tile_violations
            scores[tile_id] = tile_scores
        
        # Generate comprehensive summary
        summary = self._generate_validation_summary(
            violations, scores, coordination_validation, tiles
        )
        
        return {
            "violations": violations,
            "scores": scores,
            "coordination_validation": coordination_validation,
            "summary": summary
        }
    
    def _validate_shared_edge_matching(self, tiles: Dict[int, Image.Image]) -> Dict[str, Any]:
        """Validate shared edge matching from round-robin edge copying."""
        edge_matches = []
        edge_violations = []
        
        for edge in self.shared_edges:
            tile_a_id = edge.tile_a_id
            tile_b_id = edge.tile_b_id
            edge_type = edge.edge_type
            
            if tile_a_id not in tiles or tile_b_id not in tiles:
                continue
            
            # Convert tiles to arrays
            tile_a_array = np.array(tiles[tile_a_id].convert("L"))
            tile_b_array = np.array(tiles[tile_b_id].convert("L"))
            
            # Extract edge regions (4px width to match edge copying)
            edge_width = 4
            
            if edge_type == "right":
                # tile_a's right edge vs tile_b's left edge
                edge_a = tile_a_array[:, -edge_width:]
                edge_b = tile_b_array[:, :edge_width]
            elif edge_type == "bottom":
                # tile_a's bottom edge vs tile_b's top edge
                edge_a = tile_a_array[-edge_width:, :]
                edge_b = tile_b_array[:edge_width, :]
            else:
                continue
            
            # Calculate edge similarity (should be very high with round-robin copying)
            similarity = ssim(edge_a, edge_b, data_range=255)
            
            edge_match = {
                "tile_a_id": tile_a_id,
                "tile_b_id": tile_b_id,
                "edge_type": edge_type,
                "similarity": similarity,
                "passed": similarity >= self.edge_similarity_threshold
            }
            
            edge_matches.append(edge_match)
            
            if not edge_match["passed"]:
                edge_violations.append(
                    f"Edge mismatch between tiles {tile_a_id}-{tile_b_id} ({edge_type}): {similarity:.3f}"
                )
        
        # Calculate overall edge matching effectiveness
        if edge_matches:
            avg_similarity = np.mean([match["similarity"] for match in edge_matches])
            perfect_matches = sum(1 for match in edge_matches if match["similarity"] > 0.99)
            match_rate = perfect_matches / len(edge_matches)
        else:
            avg_similarity = 1.0
            match_rate = 1.0
        
        return {
            "edge_matches": edge_matches,
            "edge_violations": edge_violations,
            "average_similarity": avg_similarity,
            "perfect_match_rate": match_rate,
            "total_edges_validated": len(edge_matches),
            "coordination_effective": avg_similarity > 0.95  # High threshold for coordination
        }
    
    def _validate_attention_window_coherence(self, tiles: Dict[int, Image.Image]) -> Dict[str, Any]:
        """Validate cross-tile attention window coherence."""
        attention_coherence_scores = []
        coherence_violations = []
        
        # Check 32px overlap regions between adjacent tiles
        overlap_width = 32
        
        for tile_id, neighbors in self.adjacency_graph.items():
            if tile_id not in tiles:
                continue
            
            tile_array = np.array(tiles[tile_id].convert("L"))
            
            for direction, neighbor_id in neighbors.items():
                if neighbor_id not in tiles:
                    continue
                
                neighbor_array = np.array(tiles[neighbor_id].convert("L"))
                
                # Extract overlap regions
                if direction == "right":
                    # Right overlap of current tile vs left overlap of neighbor
                    current_overlap = tile_array[:, -overlap_width:]
                    neighbor_overlap = neighbor_array[:, :overlap_width]
                elif direction == "bottom":
                    # Bottom overlap of current tile vs top overlap of neighbor
                    current_overlap = tile_array[-overlap_width:, :]
                    neighbor_overlap = neighbor_array[:overlap_width, :]
                else:
                    continue
                
                # Calculate coherence (should show smooth transition from attention)
                coherence = ssim(current_overlap, neighbor_overlap, data_range=255)
                attention_coherence_scores.append(coherence)
                
                if coherence < self.attention_window_coherence_threshold:
                    coherence_violations.append(
                        f"Poor attention coherence between tiles {tile_id}-{neighbor_id} ({direction}): {coherence:.3f}"
                    )
        
        avg_coherence = np.mean(attention_coherence_scores) if attention_coherence_scores else 1.0
        
        return {
            "attention_coherence_scores": attention_coherence_scores,
            "coherence_violations": coherence_violations,
            "average_coherence": avg_coherence,
            "attention_windows_validated": len(attention_coherence_scores),
            "attention_effective": avg_coherence > self.attention_window_coherence_threshold
        }

    def _check_lighting_consistency(self, tiles: Dict[int, Image.Image]) -> float:
        """Check lighting consistency across all tiles."""
        if len(tiles) < 2:
            return 1.0

        # Calculate average brightness and contrast for each tile
        tile_lighting_stats = []

        for tile_id, tile_image in tiles.items():
            gray_array = np.array(tile_image.convert("L"))
            brightness = np.mean(gray_array)
            contrast = np.std(gray_array)
            tile_lighting_stats.append((brightness, contrast))

        # Calculate consistency (low variance = high consistency)
        brightness_values = [stats[0] for stats in tile_lighting_stats]
        contrast_values = [stats[1] for stats in tile_lighting_stats]

        brightness_consistency = 1.0 - (np.std(brightness_values) / 255.0)
        contrast_consistency = 1.0 - (np.std(contrast_values) / 255.0)

        return (brightness_consistency + contrast_consistency) / 2.0

    def _check_style_consistency(self, tiles: Dict[int, Image.Image]) -> float:
        """Check style consistency across tiles."""
        if len(tiles) < 2:
            return 1.0

        # Calculate texture features for each tile
        tile_features = []

        for tile_id, tile_image in tiles.items():
            gray_array = np.array(tile_image.convert("L"))

            # Calculate texture features using edge density and local variance
            edges = cv2.Canny(gray_array, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Local variance (texture measure)
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray_array.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray_array.astype(np.float32) - local_mean) ** 2, -1, kernel)
            avg_variance = np.mean(local_variance)

            tile_features.append((edge_density, avg_variance))

        # Calculate feature consistency
        edge_densities = [features[0] for features in tile_features]
        variances = [features[1] for features in tile_features]

        edge_consistency = 1.0 - (np.std(edge_densities) / np.mean(edge_densities)) if np.mean(edge_densities) > 0 else 1.0
        variance_consistency = 1.0 - (np.std(variances) / np.mean(variances)) if np.mean(variances) > 0 else 1.0

        return (edge_consistency + variance_consistency) / 2.0

    def _check_color_harmony(self, tiles: Dict[int, Image.Image]) -> float:
        """Check color harmony across tiles."""
        if len(tiles) < 2:
            return 1.0

        # Extract dominant colors from each tile
        tile_color_profiles = []

        for tile_id, tile_image in tiles.items():
            rgb_array = np.array(tile_image.convert("RGB"))

            # Calculate color histogram
            hist_r = cv2.calcHist([rgb_array], [0], None, [16], [0, 256])
            hist_g = cv2.calcHist([rgb_array], [1], None, [16], [0, 256])
            hist_b = cv2.calcHist([rgb_array], [2], None, [16], [0, 256])

            # Normalize histograms
            hist_r = hist_r.flatten() / np.sum(hist_r)
            hist_g = hist_g.flatten() / np.sum(hist_g)
            hist_b = hist_b.flatten() / np.sum(hist_b)

            color_profile = np.concatenate([hist_r, hist_g, hist_b])
            tile_color_profiles.append(color_profile)

        # Calculate pairwise color profile similarities
        similarities = []
        for i in range(len(tile_color_profiles)):
            for j in range(i + 1, len(tile_color_profiles)):
                # Use correlation as similarity measure
                correlation = np.corrcoef(tile_color_profiles[i], tile_color_profiles[j])[0, 1]
                if not np.isnan(correlation):
                    similarities.append(correlation)

        return np.mean(similarities) if similarities else 1.0

    def _validate_palette_compliance(self, tile_id: int, tile_image: Image.Image) -> Dict[str, Any]:
        """Validate palette compliance using actual palette from model registry."""
        violations = []
        scores = {}

        # Get target palette
        palette_name = self.config.get("palette")
        if not palette_name:
            scores["palette_deviation"] = 0.0
            return {"passed": True, "violations": [], "scores": scores}

        palette_config = self.model_registry.get_palette_config(palette_name)
        if not palette_config:
            violations.append(f"Palette '{palette_name}' not found in registry")
            return {"passed": False, "violations": violations, "scores": scores}

        # Convert tile to RGB array
        rgb_array = np.array(tile_image.convert("RGB"))

        # Calculate deviation from target palette
        target_colors = [self._hex_to_rgb(color) for color in palette_config.colors]

        # For each pixel, find closest palette color and calculate deviation
        total_deviation = 0.0
        pixel_count = rgb_array.shape[0] * rgb_array.shape[1]

        for y in range(0, rgb_array.shape[0], 4):  # Sample every 4th pixel for efficiency
            for x in range(0, rgb_array.shape[1], 4):
                pixel_color = rgb_array[y, x]

                # Find closest palette color
                min_distance = float('inf')
                for target_color in target_colors:
                    distance = euclidean(pixel_color, target_color)
                    min_distance = min(min_distance, distance)

                total_deviation += min_distance

        # Normalize deviation
        sampled_pixels = (rgb_array.shape[0] // 4) * (rgb_array.shape[1] // 4)
        avg_deviation = total_deviation / sampled_pixels if sampled_pixels > 0 else 0.0

        scores["palette_deviation"] = avg_deviation

        if avg_deviation > self.palette_deviation_threshold:
            violations.append(f"Palette deviation too high: {avg_deviation:.3f}")

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "scores": scores
        }

    def _validate_structural_compliance(self, tile_id: int, tile_image: Image.Image) -> Dict[str, Any]:
        """Validate structural compliance with ControlNet guidance."""
        violations = []
        scores = {}

        # Get tile specification
        tile_spec = self.tileset_setup.get_tile_spec(tile_id)
        if not tile_spec:
            scores["structural_compliance"] = 0.0
            violations.append("No tile specification found")
            return {"passed": False, "violations": violations, "scores": scores}

        # Analyze structural compliance
        gray_array = np.array(tile_image.convert("L"))
        height, width = gray_array.shape
        half_h, half_w = height // 2, width // 2

        quadrants = {
            "top_left": gray_array[:half_h, :half_w],
            "top_right": gray_array[:half_h, half_w:],
            "bottom_left": gray_array[half_h:, :half_w],
            "bottom_right": gray_array[half_h:, half_w:]
        }

        composition = tile_spec.structure_composition
        structural_scores = []

        for quadrant_name, quadrant_array in quadrants.items():
            expected_structure = composition.get(quadrant_name, "center")
            structure_score = self._analyze_quadrant_structure(quadrant_array, expected_structure)
            structural_scores.append(structure_score)
            scores[f"structure_{quadrant_name}"] = structure_score

            if structure_score < self.structural_compliance_threshold:
                violations.append(f"Quadrant {quadrant_name} doesn't match expected structure {expected_structure}")

        overall_compliance = np.mean(structural_scores) if structural_scores else 0.0
        scores["structural_compliance"] = overall_compliance

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "scores": scores
        }

    def _validate_sub_tile_coherence(self, tile_id: int, tile_image: Image.Image) -> Dict[str, Any]:
        """Validate coherence between sub-tiles."""
        violations = []
        scores = {}

        # Convert to array and extract quadrants
        img_array = np.array(tile_image.convert("L"))
        height, width = img_array.shape
        mid_h, mid_w = height // 2, width // 2

        top_left = img_array[:mid_h, :mid_w]
        top_right = img_array[:mid_h, mid_w:]
        bottom_left = img_array[mid_h:, :mid_w]
        bottom_right = img_array[mid_h:, mid_w:]

        # Check boundary coherence
        coherence_scores = []

        # Boundaries between quadrants
        boundaries = [
            ("top", top_left[:, -3:], top_right[:, :3]),
            ("left", top_left[-3:, :], bottom_left[:3, :]),
            ("right", top_right[-3:, :], bottom_right[:3, :]),
            ("bottom", bottom_left[:, -3:], bottom_right[:, :3])
        ]

        for boundary_name, region_a, region_b in boundaries:
            coherence = ssim(region_a, region_b, data_range=255)
            coherence_scores.append(coherence)
            scores[f"coherence_{boundary_name}_boundary"] = coherence

        overall_coherence = np.mean(coherence_scores)
        scores["sub_tile_coherence"] = overall_coherence

        if overall_coherence < self.sub_tile_coherence_threshold:
            violations.append(f"Sub-tile coherence too low: {overall_coherence:.3f}")

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "scores": scores
        }

    def _analyze_quadrant_structure(self, quadrant_array: np.ndarray, expected_structure: str) -> float:
        """Analyze quadrant for expected structural features."""
        edges = cv2.Canny(quadrant_array, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Expected edge densities for different structures
        expected_densities = {
            "center": 0.1, "border_top": 0.3, "border_right": 0.3, "border_bottom": 0.3, "border_left": 0.3,
            "edge_ne": 0.4, "edge_nw": 0.4, "edge_se": 0.4, "edge_sw": 0.4,
            "corner_ne": 0.5, "corner_nw": 0.5, "corner_se": 0.5, "corner_sw": 0.5,
        }

        expected_density = expected_densities.get(expected_structure, 0.2)
        density_diff = abs(edge_density - expected_density)
        return max(0.0, 1.0 - (density_diff / 0.5))

    def _generate_validation_summary(self, violations: Dict[int, List[str]],
                                   scores: Dict[int, Dict[str, float]],
                                   coordination_validation: Dict[str, Any],
                                   tiles: Dict[int, Image.Image]) -> Dict[str, Any]:
        """Generate comprehensive validation summary."""
        total_violations = sum(len(v) for v in violations.values())
        tiles_with_violations = len([v for v in violations.values() if v])

        # Coordination effectiveness
        edge_effective = coordination_validation["shared_edge_matching"]["coordination_effective"]
        attention_effective = coordination_validation["attention_window_coherence"]["attention_effective"]
        consistency_effective = coordination_validation["global_consistency"]["simultaneous_generation_effective"]

        coordination_effectiveness = (edge_effective + attention_effective + consistency_effective) / 3.0

        return {
            "total_tiles": len(tiles),
            "tiles_with_violations": tiles_with_violations,
            "total_violations": total_violations,
            "violation_rate": tiles_with_violations / len(tiles) if tiles else 0,
            "edge_violations": len(coordination_validation["shared_edge_matching"]["edge_violations"]),
            "coordination_effectiveness": coordination_effectiveness,
            "multi_tile_system_working": coordination_effectiveness > 0.8,
            "average_edge_similarity": coordination_validation["shared_edge_matching"]["average_similarity"],
            "average_attention_coherence": coordination_validation["attention_window_coherence"]["average_coherence"],
            "global_consistency_score": coordination_validation["global_consistency"]["average_consistency"]
        }

    def make_regeneration_decisions(self, validation_result: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Make intelligent regeneration decisions based on validation results."""
        decisions = {}

        # If multi-tile coordination is working well, be more lenient
        coordination_working = validation_result["summary"]["multi_tile_system_working"]

        for tile_id, tile_violations in validation_result["violations"].items():
            should_regenerate = False
            reasons = []

            # Critical: Edge matching failures (indicates coordination system failure)
            edge_violations = [v for v in tile_violations if "Edge mismatch" in v]
            if edge_violations:
                should_regenerate = True
                reasons.extend(edge_violations)

            # If coordination is working, be more forgiving of other violations
            if coordination_working:
                # Only regenerate for severe violations
                severe_violations = [v for v in tile_violations if "too low" in v and "0.5" in v]
                if severe_violations:
                    should_regenerate = True
                    reasons.extend(severe_violations)
            else:
                # If coordination isn't working, regenerate for any violations
                if tile_violations:
                    should_regenerate = True
                    reasons.extend(tile_violations)

            decisions[tile_id] = {
                "regenerate": should_regenerate,
                "reason": "; ".join(reasons) if reasons else "No violations",
                "coordination_working": coordination_working
            }

        return decisions

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _validate_global_consistency(self, tiles: Dict[int, Image.Image]) -> Dict[str, Any]:
        """Validate global consistency from simultaneous generation."""
        consistency_scores = []
        consistency_violations = []
        
        # Check lighting consistency across tiles
        lighting_consistency = self._check_lighting_consistency(tiles)
        consistency_scores.append(lighting_consistency)
        
        # Check style consistency
        style_consistency = self._check_style_consistency(tiles)
        consistency_scores.append(style_consistency)
        
        # Check color harmony
        color_harmony = self._check_color_harmony(tiles)
        consistency_scores.append(color_harmony)
        
        avg_consistency = np.mean(consistency_scores)
        
        if lighting_consistency < self.cross_tile_consistency_threshold:
            consistency_violations.append(f"Poor lighting consistency: {lighting_consistency:.3f}")
        
        if style_consistency < self.cross_tile_consistency_threshold:
            consistency_violations.append(f"Poor style consistency: {style_consistency:.3f}")
        
        if color_harmony < self.cross_tile_consistency_threshold:
            consistency_violations.append(f"Poor color harmony: {color_harmony:.3f}")
        
        return {
            "lighting_consistency": lighting_consistency,
            "style_consistency": style_consistency,
            "color_harmony": color_harmony,
            "average_consistency": avg_consistency,
            "consistency_violations": consistency_violations,
            "simultaneous_generation_effective": avg_consistency > self.cross_tile_consistency_threshold
        }
