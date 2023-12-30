import torch

from ....sgm.modules.diffusionmodules.discretizer import Discretization


class Img2ImgDiscretizationWrapper:
    """
    обертывает дискретизатор и обрезает сигмы
    params:
        сила вклада обработки: float между 0.0 и 1.0. 1.0 означает полное семплирование (возвращаются все сигмы)
    """
    def __init__(self, discretization: Discretization, strength: float = 1.0):
        self.discretization = discretization
        self.strength = strength
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        #сигмы сначала начинаются большими, а потом уменьшаются
        sigmas = self.discretization(*args, **kwargs)
        print(f"Сигмы после дискретизации, до обрезки img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        sigmas = sigmas[: max(int(self.strength * len(sigmas)), 1)]
        print("Индекс обрезки: ", max(int(self.strength * len(sigmas)), 1))
        sigmas = torch.flip(sigmas, (0,))
        print(f"Сигмы после обрезки: ", sigmas)
        return sigmas

class Txt2NoisyDiscretizationWrapper:
    """
    обертывает дискретизатор и обрезает сигмы
    params:
        сила вклада: float между 0.0 и 1.0. 0.0 означает полное семплирование (возвращаются все сигмы)
    """
    def __init__(self, discretization: Discretization, strength: float = 0.0, original_steps = None):
        self.discretization = discretization
        self.strength = strength
        self.original_steps = original_steps
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        #сигмы сначала начинаются большими, а потом уменьшаются
        sigmas = self.discretization(*args, **kwargs)
        print(f"Сигмы после дискретизации, до обрезки img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        if self.original_steps is None:
            steps = len(sigmas)
        else:
            steps = self.original_steps + 1
        prune_index = max(min(int(self.strength * steps) - 1, steps - 1), 0)
        sigmas = sigmas[prune_index:]
        print("Индекс обрезки: ", prune_index)
        sigmas = torch.flip(sigmas, (0,))
        print(f"Сигмы после обрезки: ", sigmas)
        return sigmas