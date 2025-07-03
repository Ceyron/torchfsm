from .operator import (
    Operator,
    Convection,
    Laplacian,
    Biharmonic,
    KSConvection,
    SpatialDerivative,
    VorticityConvection,
    NSPressureConvection,
    ImplicitSource,
    HyperDiffusion,
    ChannelWisedDiffusion,
    GrayScottSource,
)
from typing import Optional, Union
from torch import Tensor

def Burgers(nu: float) -> Operator:
    r"""
    Burgers equation:
        $$\frac{\partial \mathbf{u}}{\partial t} =-\mathbf{u} \cdot \nabla \mathbf{u} + \nu \nabla^2 \mathbf{u}$$

    Args:
        nu (float): Viscosity coefficient.    
        
    Returns:
        Operator: The operator representing the Burgers equation.

    """
    return nu * Laplacian() - Convection()

def KuramotoSivashinsky() -> Operator:
    r"""
    1D Kuramoto-Sivashinsky equation:
        $$\frac{\partial \phi}{\partial t}=-\frac{\partial^2 \phi}{\partial x^2} -\frac{\partial^4 \phi}{\partial x^4} - \phi\frac{\partial\phi}{\partial x}$$
    
    Returns:
        Operator: The operator representing the Kuramoto-Sivashinsky equation.
    """
    ks_eqn = -Laplacian() - Biharmonic() - Convection()
    ks_eqn.register_additional_check(
        lambda dim_value, dim_mesh: dim_value == 1 and dim_mesh == 1
    )
    return ks_eqn

def KuramotoSivashinskyHighDim() -> Operator:
    r"""
    High dimensional Kuramoto-Sivashinsky equation:
        $$\frac{\partial \mathbf{\phi}}{\partial t}=-\nabla^2 \phi- \nabla^4 \phi - \frac{1}{2}|\nabla \phi|^2$$
    
    Returns:
        Operator: The operator representing the Kuramoto-Sivashinsky equation.
    """
    return -Laplacian() - Biharmonic() - KSConvection()

def KortewegDeVries(dispersion_coef=1, convection_coef: float = 6.0) -> Operator:
    r"""
    Korteweg-De Vries equation:
        $$\frac{\partial \phi}{\partial t}=-c_1\frac{\partial^3 \phi}{\partial x^3} + c_2 \phi\frac{\partial\phi}{\partial x}$$

    Args:
        dispersion_coef (float): Dispersion coefficient. Default is 1.
        convection_coef (float): Convection coefficient. Default is 6.0.

    Returns:
        Operator: The operator representing the Korteweg-De Vries equation.    

    """
    return -dispersion_coef * SpatialDerivative(0, 3) + convection_coef * Convection()

def NavierStokesVorticity(Re:Union[float,Tensor],force:Optional[Operator]=None)->Operator:
    r"""
    Navier-Stokes equation in vorticity form:
        $$\frac{\partial \omega}{\partial t} + (\mathbf{u}\cdot\nabla) \omega = \frac{1}{Re} \nabla^2 \omega + \nabla \times \mathbf{f}$$
    
    Args:
        Re (float): Reynolds number.
        force (Optional[Operator]): Optional external force term. Default is None.
            If provided, it will be added to the vorticity equation. Note that the provided force should be $\nabla \times \mathbf{f}$ rather than $\mathbf{f}$ itself.

    Returns:
        Operator: The operator representing the Navier-Stokes equation in vorticity form.

    """
    ns_vorticity=-VorticityConvection() + 1/Re*Laplacian()
    if force is not None:
        ns_vorticity+=force
    return ns_vorticity

def NavierStokes(Re:Union[float,Tensor],force:Optional[Operator]=None)->Operator:
    r"""
    Navier-Stokes equation:
        $$\frac{\partial\mathbf{u}}{\partial t}=-\nabla (\nabla^{-2} \nabla \cdot (\left(\mathbf{u}\cdot\nabla\right)\mathbf{u}-f))-\left(\mathbf{u}\cdot\nabla\right)\mathbf{u}+\nu \nabla^2 \mathbf{u} + \mathbf{f}$$
    
    Args:
        Re (float): Reynolds number.
        force (Optional[Operator]): Optional external force term. Default is None.
            If provided, it will be added to the vorticity equation. 

    Returns:
        Operator: The operator representing the Navier-Stokes equation.

    """

    return NSPressureConvection(force)+1/Re*Laplacian()

def KPPFisher(nu:Union[float,Tensor],r:Union[float,Tensor]) -> Operator:
    r"""
    Fisher-KPP equation:
        $$\frac{\partial u}{\partial t} = r u (1 - u) + \nu \nabla^2 u$$
    
    Args:
        nu (Union[float,Tensor]): Diffusion coefficient.
        r (Union[float,Tensor]): Growth rate.

    Returns:
        Operator: The operator representing the Fisher-KPP equation.
    
    """
    return nu*Laplacian()+r*ImplicitSource(lambda phi:phi*(1-phi))

def SwiftHohenberg(r:float) -> Operator:
    r"""
    Swift-Hohenberg equation:
        $$\frac{\partial \phi}{\partial t} = \nu \nabla^2 \phi + r \phi (1 - \phi)$$

    Args:
        r (float): Control parameter.

    Returns:
        Operator: The operator representing the Swift-Hohenberg equation.
    """
    return -HyperDiffusion()-2*Laplacian()+ImplicitSource(lambda phi:r*phi-phi+phi**2-phi**3)

def GrayScott(nu_0: float, 
              nu_1:float, 
              feed_rate:float, 
              kill_rate:float, 
              de_aliasing_rate=0.5) -> Operator:
    r"""
    Gray-Scott equation:
        $$\begin{aligned}
        \frac{\partial \phi_0}{\partial t} &= \nu_0 \Delta \phi_0 - \phi_0 \phi_1^2 + f (1 - \phi_0) \\
        \frac{\partial \phi_1}{\partial t} &= \nu_1 \Delta \phi_1 + \phi_0 \phi_1^2 - (f + k) \phi_1
        \end{aligned}
        $$

    Args:
        nu_0 (float): Diffusion coefficient for \(\phi_0\).
        nu_1 (float): Diffusion coefficient for \(\phi_1\).
        feed_rate (float): Feed rate \(f\).
        kill_rate (float): Kill rate \(k\).
        de_aliasing_rate (float): De-aliasing rate for the operator. Default is 0.5 due to the cubic nonlinearity.

    Returns:
        Operator: The operator representing the Gray-Scott equation.
    """

    op = ChannelWisedDiffusion([nu_0,nu_1])+ GrayScottSource(feed_rate, kill_rate)
    op.set_de_aliasing_rate(de_aliasing_rate) # GrayScottSource requires lower de-aliasing rate due to the cubic nonlinearity
    return op