#include "common.h"
#include <cstdio>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/ext/quaternion_float.hpp>
#include <glm/gtx/quaternion.hpp>

namespace gaussian_utils {

namespace {

	// Useful note:
	// glm uses column major matrix layout.
	// Everything else uses row major matrix layout.
	// We need to transpose as needed.

	template <typename T>
	__device__ constexpr unsigned capToWarpSize(T val)
	{
		return val <= 32 ? val : 32;
	}

	__global__ void nearestCenter3D_ForwardImpl(
		size_t N_points,
		size_t N_centers,
		const glm::vec3* in_points,
		const glm::vec3* in_centers,
		glm::vec3* out_vectors,
		size_t* out_indices
	)
	{
		auto point_idx = cg::this_grid().thread_rank();
		bool is_active = point_idx < N_points;

		glm::vec3 point;
		if (is_active) {
			point = in_points[point_idx];
		}

		size_t best_idx;
		glm::vec3 best_vector;
		float best_sqdist;

		for (size_t base = 0; base < N_centers; base += 32) {
			unsigned maxfetch = capToWarpSize(N_centers - base);

			glm::vec3 warpcenter;
			unsigned lane = threadIdx.x & 0x1f;
			if (lane < maxfetch) {
				warpcenter = in_centers[base+lane];
			}

			for (unsigned i = 0; i < maxfetch; i ++) {
				glm::vec3 center;
				for (unsigned row = 0; row < 3; row ++) {
					center[row] = __shfl_sync(UINT32_MAX, warpcenter[row], i);
				}

				if (!is_active) {
					continue;
				}

				size_t curidx = base+i;
				glm::vec3 vector = point - center;
				float sqdist = glm::dot(vector, vector);

				if (curidx == 0 || sqdist < best_sqdist) {
					best_idx = curidx;
					best_vector = vector;
					best_sqdist = sqdist;
				}
			}

		}

		if (is_active) {
			out_vectors[point_idx] = best_vector;
			out_indices[point_idx] = best_idx;
		}
	}

	__global__ void nearestCenter3D_BackwardImpl(
		size_t N_points,
		size_t N_centers,
		const glm::vec3* in_dL_dvectors,
		const size_t* saved_indices,
		glm::vec3* out_dL_dpoints,
		glm::vec3* out_dL_dcenters
	)
	{
		auto point_idx = cg::this_grid().thread_rank();
		if (point_idx >= N_points) {
			return;
		}

		size_t center_idx = saved_indices[point_idx];
		auto& in_dL_dvector = in_dL_dvectors[point_idx];
		auto& out_dL_dcenter = out_dL_dcenters[center_idx];

		out_dL_dpoints[point_idx] = in_dL_dvector;

		for (unsigned row = 0; row < 3; row ++) {
			atomicAdd(&out_dL_dcenter[row], -in_dL_dvector[row]);
		}
	}

	__global__ void nearestGaussian3D_ForwardImpl(
		size_t N_points,
		size_t N_gaussians,
		const glm::vec3* in_points,
		const glm::vec3* in_centers,
		const glm::mat3* in_matrices,
		float* out_sqmahal,
		size_t* out_indices
	)
	{
		auto point_idx = cg::this_grid().thread_rank();
		bool is_active = point_idx < N_points;

		glm::vec3 point;
		if (is_active) {
			point = in_points[point_idx];
		}

		size_t best_idx;
		float best_sqmahal;

		for (size_t base = 0; base < N_gaussians; base += 32) {
			unsigned maxfetch = capToWarpSize(N_gaussians - base);

			glm::vec3 warpcenter;
			glm::mat3 warpmat;
			unsigned lane = threadIdx.x & 0x1f;
			if (lane < maxfetch) {
				warpcenter = in_centers[base+lane];
				warpmat = in_matrices[base+lane];
			}

			for (unsigned i = 0; i < maxfetch; i ++) {
				glm::vec3 gcenter;
				glm::mat3 gmat;
				for (unsigned col = 0; col < 3; col ++) {
					gcenter[col] = __shfl_sync(UINT32_MAX, warpcenter[col], i);
					for (unsigned row = 0; row < 3; row ++) {
						gmat[col][row] = __shfl_sync(UINT32_MAX, warpmat[col][row], i);
					}
				}

				if (!is_active) {
					continue;
				}

				size_t curidx = base+i;
				glm::vec3 vector = (point - gcenter) * gmat; // Mv = v'M' (row/column major swap)
				float sqmahal = glm::dot(vector, vector);

				if (curidx == 0 || sqmahal < best_sqmahal) {
					best_idx = curidx;
					best_sqmahal = sqmahal;
				}
			}

		}

		if (is_active) {
			out_sqmahal[point_idx] = best_sqmahal;
			out_indices[point_idx] = best_idx;
		}
	}

	__global__ void nearestGaussian3D_BackwardImpl(
		size_t N_points,
		size_t N_gaussians,
		const float* in_dL_dsqmahal,
		const glm::vec3* saved_points,
		const glm::vec3* saved_centers,
		const glm::mat3* saved_matrices,
		const size_t* saved_indices,
		glm::vec3* out_dL_dpoints,
		glm::vec3* out_dL_dcenters,
		glm::mat3* out_dL_dmatrices
	)
	{
		auto point_idx = cg::this_grid().thread_rank();
		if (point_idx >= N_points) {
			return;
		}

		auto& point = saved_points[point_idx];
		size_t gaussian_idx = saved_indices[point_idx];
		auto& center = saved_centers[gaussian_idx];
		auto& matrix = saved_matrices[gaussian_idx];
		auto vector = (point - center) * matrix; // see above (implicit transpose)

		auto dL_dvector = 2*in_dL_dsqmahal[point_idx]*vector;
		auto& dL_dcenter = out_dL_dcenters[gaussian_idx];
		auto& dL_dmatrix = out_dL_dmatrices[gaussian_idx];

		out_dL_dpoints[point_idx] = matrix * dL_dvector;

		if (!out_dL_dcenters || !out_dL_dmatrices) {
			return;
		}

		for (unsigned row = 0; row < 3; row ++) {
			atomicAdd(&dL_dcenter[row], -dL_dvector[row]);

			for (unsigned col = 0; col < 3; col ++) {
				// Sanity preserving note: First index is always major in glm.
				// In essence, this is element (first*3 + second) in memory.
				// We consider the first index to be the rows, ignoring glm's convention.
				// Therefore this code is correct.
				atomicAdd(&dL_dmatrix[row][col], dL_dvector[row] * point[col]);
			}
		}
	}

	__global__ void indexedTransform3D_ForwardImpl(
		size_t N_points,
		const glm::vec3* in_points,
		const glm::mat3* in_matrices,
		const size_t* in_indices,
		glm::vec3* out_points
	)
	{
		auto point_idx = cg::this_grid().thread_rank();
		if (point_idx >= N_points) {
			return;
		}

		auto& in_point = in_points[point_idx];
		auto in_matrix = in_matrices[in_indices[point_idx]]; // implicit transposition

		out_points[point_idx] = in_point * in_matrix; // in effect (Mp)' = p'M'
	}

	__global__ void indexedTransform3D_BackwardImpl(
		size_t N_points,
		const glm::vec3* in_dL_doutpoints,
		const glm::vec3* saved_inpoints,
		const glm::mat3* saved_matrices,
		const size_t* saved_indices,
		glm::vec3* out_dL_dinpoints,
		glm::mat3* out_dL_dmatrices
	)
	{
		auto point_idx = cg::this_grid().thread_rank();
		if (point_idx >= N_points) {
			return;
		}

		auto& dL_dout = in_dL_doutpoints[point_idx];
		auto& point = saved_inpoints[point_idx];
		auto& matrix_transpose = saved_matrices[saved_indices[point_idx]];
		auto& dL_dmatrix = out_dL_dmatrices[saved_indices[point_idx]];

		out_dL_dinpoints[point_idx] = matrix_transpose * dL_dout;

		for (unsigned row = 0; row < 3; row ++) {
			for (unsigned col = 0; col < 3; col ++) {
				// Sanity preserving note: First index is always major in glm.
				// In essence, this is element (first*3 + second) in memory.
				// We consider the first index to be the rows, ignoring glm's convention.
				// Therefore this code is correct.
				atomicAdd(&dL_dmatrix[row][col], dL_dout[row] * point[col]);
			}
		}

	}

}

void nearestCenter3D_Forward(
	size_t N_points,
	size_t N_centers,
	const float* in_points,
	const float* in_centers,
	float* out_vectors,
	size_t* out_indices
)
{
	nearestCenter3D_ForwardImpl<<<(N_points + UTIL_NUM_THREADS - 1) / UTIL_NUM_THREADS, UTIL_NUM_THREADS>>>(
		N_points,
		N_centers,
		(const glm::vec3*)in_points,
		(const glm::vec3*)in_centers,
		(glm::vec3*)out_vectors,
		out_indices
	);
}

void nearestCenter3D_Backward(
	size_t N_points,
	size_t N_centers,
	const float* in_dL_dvectors,
	const size_t* saved_indices,
	float* out_dL_dpoints,
	float* out_dL_dcenters
)
{
	nearestCenter3D_BackwardImpl<<<(N_points + UTIL_NUM_THREADS - 1) / UTIL_NUM_THREADS, UTIL_NUM_THREADS>>>(
		N_points,
		N_centers,
		(const glm::vec3*)in_dL_dvectors,
		saved_indices,
		(glm::vec3*)out_dL_dpoints,
		(glm::vec3*)out_dL_dcenters
	);
}

void nearestGaussian3D_Forward(
	size_t N_points,
	size_t N_gaussians,
	const float* in_points,
	const float* in_centers,
	const float* in_matrices,
	float* out_sqmahal,
	size_t* out_indices
)
{
	nearestGaussian3D_ForwardImpl<<<(N_points + UTIL_NUM_THREADS - 1) / UTIL_NUM_THREADS, UTIL_NUM_THREADS>>>(
		N_points,
		N_gaussians,
		(const glm::vec3*)in_points,
		(const glm::vec3*)in_centers,
		(const glm::mat3*)in_matrices,
		out_sqmahal,
		out_indices
	);
}

void nearestGaussian3D_Backward(
	size_t N_points,
	size_t N_gaussians,
	const float* in_dL_dsqmahal,
	const float* saved_points,
	const float* saved_centers,
	const float* saved_matrices,
	const size_t* saved_indices,
	float* out_dL_dpoints,
	float* out_dL_dcenters,
	float* out_dL_dmatrices
)
{
	nearestGaussian3D_BackwardImpl<<<(N_points + UTIL_NUM_THREADS - 1) / UTIL_NUM_THREADS, UTIL_NUM_THREADS>>>(
		N_points,
		N_gaussians,
		in_dL_dsqmahal,
		(const glm::vec3*)saved_points,
		(const glm::vec3*)saved_centers,
		(const glm::mat3*)saved_matrices,
		saved_indices,
		(glm::vec3*)out_dL_dpoints,
		(glm::vec3*)out_dL_dcenters,
		(glm::mat3*)out_dL_dmatrices
	);
}

void indexedTransform3D_Forward(
	size_t N_points,
	const float* in_points,
	const float* in_matrices,
	const size_t* in_indices,
	float* out_points
)
{
	indexedTransform3D_ForwardImpl<<<(N_points + UTIL_NUM_THREADS - 1) / UTIL_NUM_THREADS, UTIL_NUM_THREADS>>>(
		N_points,
		(const glm::vec3*)in_points,
		(const glm::mat3*)in_matrices,
		in_indices,
		(glm::vec3*)out_points
	);
}

void indexedTransform3D_Backward(
	size_t N_points,
	const float* in_dL_doutpoints,
	const float* saved_inpoints,
	const float* saved_matrices,
	const size_t* saved_indices,
	float* out_dL_dinpoints,
	float* out_dL_dmatrices
)
{
	indexedTransform3D_BackwardImpl<<<(N_points + UTIL_NUM_THREADS - 1) / UTIL_NUM_THREADS, UTIL_NUM_THREADS>>>(
		N_points,
		(const glm::vec3*)in_dL_doutpoints,
		(const glm::vec3*)saved_inpoints,
		(const glm::mat3*)saved_matrices,
		saved_indices,
		(glm::vec3*)out_dL_dinpoints,
		(glm::mat3*)out_dL_dmatrices
	);
}

}
