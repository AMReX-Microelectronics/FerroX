#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>
#include <cmath>
#include "DerivativeAlgorithm.H"
#include "Energy_Calculation.H"

using namespace amrex;

Real ComputeLandauEnergy(Array<MultiFab, AMREX_SPACEDIM>& P,
                         MultiFab& MaterialMask,
                         const Geometry& geom,
                         Real alpha,
                         Real beta,
                         Real gamma_coeff)
{
    const Real dV = geom.CellSize(0) * geom.CellSize(1) * geom.CellSize(2);

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(P[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const& px = P[0].array(mfi);
        auto const& py = P[1].array(mfi);
        auto const& pz = P[2].array(mfi);
        auto const& mask = MaterialMask.array(mfi);

        reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
                if (mask(i,j,k) == 0.0) {
                    Real P2 = px(i,j,k)*px(i,j,k) + py(i,j,k)*py(i,j,k) + pz(i,j,k)*pz(i,j,k);
                    Real P4 = P2 * P2;
                    Real P6 = P4 * P2;
                    Real f_landau = 0.5 * alpha * P2 + 0.25 * beta * P4 + (1.0/6.0) * gamma_coeff * P6;
                    return {f_landau * dV};
                } else {
                    return {0.0};
                }
            });
    }

    Real landau_energy = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealSum(landau_energy);
    return landau_energy;
}


Real ComputeGradientEnergy(const Array<MultiFab, AMREX_SPACEDIM>& P,
                           const MultiFab& MaterialMask,
                           const Geometry& geom,
                           Real g11,
                           Real g44)
{
    const Real dV = geom.CellSize(0) * geom.CellSize(1) * geom.CellSize(2);
    const auto dx = geom.CellSizeArray();  // GpuArray<Real, 3>

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(P[2], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const& pz = P[2].array(mfi);  // Scalar polarization (Pz)
        auto const& mask = MaterialMask.array(mfi);

        reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
                if (mask(i,j,k) == 0.0) {
                    Real dPdx = DPDx(pz, mask, i, j, k, dx);
                    Real dPdy = DPDy(pz, mask, i, j, k, dx);
                    Real dPdz = DPDz(pz, mask, i, j, k, dx);

                    Real f_grad = 0.5 * (g44 * (dPdx*dPdx + dPdy*dPdy) + g11 * dPdz*dPdz);
                    return {f_grad * dV};
                } else {
                    return {0.0};
                }
            });
    }

    Real grad_energy = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealSum(grad_energy);
    return grad_energy;
}



Real ComputeElectrostaticEnergy( Array<MultiFab, AMREX_SPACEDIM>& P,
                                 Array<MultiFab, AMREX_SPACEDIM>& E,
                                const MultiFab& MaterialMask,
                                const Geometry& geom)
{
    const Real dV = geom.CellSize(0) * geom.CellSize(1) * geom.CellSize(2);

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(P[2], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const& pz = P[2].array(mfi);  // scalar polarization: P = Pz
        auto const& ez = E[2].array(mfi);  // scalar electric field: E = Ez
        auto const& mask = MaterialMask.array(mfi);

        reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
                if (mask(i,j,k) == 0.0) {
                    Real f_elec = ez(i,j,k) * pz(i,j,k);  // E · P
                    return {f_elec * dV};
                } else {
                    return {0.0};
                }
            });
    }

    Real electrostatic_energy = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealSum(electrostatic_energy);
    return electrostatic_energy;
}