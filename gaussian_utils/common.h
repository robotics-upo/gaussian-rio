#pragma once
#include <torch/extension.h>

#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#endif

#define GLM_FORCE_QUAT_DATA_WXYZ
#include <glm/glm.hpp>

#define UTIL_NUM_THREADS 256

namespace gaussian_utils {

	void rotScale3D_Forward(
		size_t N,
		const float* in_scales,
		const float* in_quats,
		float* out_matrices,
		bool scale_first
	);

	void rotScale3D_Backward(
		size_t N,
		const float* in_dL_dM,
		const float* saved_scales,
		const float* saved_quats,
		float* out_dL_dscales,
		float* out_dL_dquats,
		bool scale_first
	);

	void nearestCenter3D_Forward(
		size_t N_points,
		size_t N_centers,
		const float* in_points,
		const float* in_centers,
		float* out_vectors,
		size_t* out_indices
	);

	void nearestCenter3D_Backward(
		size_t N_points,
		size_t N_centers,
		const float* in_dL_dvectors,
		const size_t* saved_indices,
		float* out_dL_dpoints,
		float* out_dL_dcenters
	);

	void nearestGaussian3D_Forward(
		size_t N_points,
		size_t N_gaussians,
		const float* in_points,
		const float* in_centers,
		const float* in_matrices,
		float* out_sqmahal,
		size_t* out_indices
	);

	void nearestGaussian3D_Backward(
		size_t N_points,
		size_t N_gaussians,
		const float* in_dL_dsqmahal,
		const float* saved_points,
		const float* saved_centers,
		const float* saved_matrices,
		const size_t* saved_indices,
		float* out_dL_dpoints,
		float* out_dL_dcenters = nullptr,
		float* out_dL_dmatrices = nullptr
	);

	void indexedTransform3D_Forward(
		size_t N_points,
		const float* in_points,
		const float* in_matrices,
		const size_t* in_indices,
		float* out_points
	);

	void indexedTransform3D_Backward(
		size_t N_points,
		const float* in_dL_doutpoints,
		const float* saved_inpoints,
		const float* saved_matrices,
		const size_t* saved_indices,
		float* out_dL_dinpoints,
		float* out_dL_dmatrices
	);

}
