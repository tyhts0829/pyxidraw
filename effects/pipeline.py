from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import BaseEffect


class EffectPipeline:
    """Pipeline for chaining multiple effects together."""
    
    def __init__(self, effects: Sequence[BaseEffect] | None = None):
        """Initialize pipeline with optional initial effects.
        
        Args:
            effects: Sequence of effects to apply in order
        """
        self.effects = list(effects) if effects else []
    
    def add(self, effect: BaseEffect) -> EffectPipeline:
        """Add an effect to the pipeline.
        
        Args:
            effect: Effect to add
            
        Returns:
            Self for method chaining
        """
        self.effects.append(effect)
        return self
    
    def remove(self, effect: BaseEffect) -> EffectPipeline:
        """Remove an effect from the pipeline.
        
        Args:
            effect: Effect to remove
            
        Returns:
            Self for method chaining
        """
        if effect in self.effects:
            self.effects.remove(effect)
        return self
    
    def clear(self) -> EffectPipeline:
        """Clear all effects from the pipeline.
        
        Returns:
            Self for method chaining
        """
        self.effects.clear()
        return self
    
    def apply(self, vertices_list: list[np.ndarray], **params: Any) -> list[np.ndarray]:
        """Apply all effects in the pipeline sequentially.
        
        Args:
            vertices_list: Input vertex arrays
            **params: Parameters passed to all effects
            
        Returns:
            Transformed vertex arrays
        """
        result = vertices_list
        for effect in self.effects:
            result = effect(result, **params)
        return result
    
    def __call__(self, vertices_list: list[np.ndarray], **params: Any) -> list[np.ndarray]:
        """Apply the pipeline (alias for apply method)."""
        return self.apply(vertices_list, **params)
    
    def __len__(self) -> int:
        """Get number of effects in pipeline."""
        return len(self.effects)
    
    def __getitem__(self, index: int) -> BaseEffect:
        """Get effect at index."""
        return self.effects[index]
    
    def __iter__(self):
        """Iterate over effects."""
        return iter(self.effects)
    
    def clear_all_caches(self):
        """Clear caches for all effects in the pipeline."""
        for effect in self.effects:
            effect.clear_cache()