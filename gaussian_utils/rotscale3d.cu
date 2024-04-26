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

	__global__ void rotScale3D_ForwardImpl(
		size_t N,
		const glm::vec3* in_scales,
		const glm::quat* in_quats,
		glm::mat3* out_matrices,
		bool scale_first
	)
	{
		auto idx = cg::this_grid().thread_rank();
		if (idx >= N) {
			return;
		}

		const auto& s = in_scales[idx];
		const auto& q = in_quats[idx];

		auto M = glm::mat3(q);
		if (scale_first) {
			// M = R*S
			M[0] *= s.x;
			M[1] *= s.y;
			M[2] *= s.z;
		} else {
			// M = S*R
			M[0] *= s;
			M[1] *= s;
			M[2] *= s;
		}

		// Store matrix as row-major
		out_matrices[idx] = glm::transpose(M);
	}

	__global__ void rotScale3D_BackwardImpl(
		size_t N,
		const glm::mat3* in_dL_dMt,
		const glm::vec3* saved_scales,
		const glm::quat* saved_quats,
		glm::vec3* out_dL_dscales,
		glm::quat* out_dL_dquats,
		bool scale_first
	)
	{
		auto idx = cg::this_grid().thread_rank();
		if (idx >= N) {
			return;
		}

		const auto& dL_dMt = in_dL_dMt[idx]; // ROW MAJOR = transposed COLUMN MAJOR
		auto dL_dM = glm::transpose(dL_dMt);
		auto& dL_dscale = out_dL_dscales[idx];
		auto& dL_dquat = out_dL_dquats[idx];

		const auto& s = saved_scales[idx];
		const auto& q = saved_quats[idx];
		auto R = glm::mat3(q);

		// dL/dS = dL/dM dM/dS
		if (scale_first) {
			dL_dscale.x = glm::dot(dL_dM[0], R[0]);
			dL_dscale.y = glm::dot(dL_dM[1], R[1]);
			dL_dscale.z = glm::dot(dL_dM[2], R[2]);
		} else {
			auto Rt = glm::transpose(R);

			dL_dscale.x = glm::dot(dL_dMt[0], Rt[0]);
			dL_dscale.y = glm::dot(dL_dMt[1], Rt[1]);
			dL_dscale.z = glm::dot(dL_dMt[2], Rt[2]);
		}

		// dL/dR = dL/dM dM/dR
		auto& dL_dR = dL_dM;
		if (scale_first) {
			dL_dR[0] *= s.x;
			dL_dR[1] *= s.y;
			dL_dR[2] *= s.z;
		} else {
			dL_dR[0] *= s;
			dL_dR[1] *= s;
			dL_dR[2] *= s;
		}

		// dL/dq = dL/dR dR/dq
		dL_dquat.w =   2*q.x*(dL_dR[1][2]-dL_dR[2][1]) + 2*q.y*(dL_dR[2][0]-dL_dR[0][2]) + 2*q.z*(dL_dR[0][1]-dL_dR[1][0]);
		dL_dquat.x = - 4*q.x*(dL_dR[1][1]+dL_dR[2][2]) + 2*q.y*(dL_dR[0][1]+dL_dR[1][0]) + 2*q.z*(dL_dR[0][2]+dL_dR[2][0]) + 2*q.w*(dL_dR[1][2]-dL_dR[2][1]);
		dL_dquat.y =   2*q.x*(dL_dR[0][1]+dL_dR[1][0]) - 4*q.y*(dL_dR[0][0]+dL_dR[2][2]) + 2*q.z*(dL_dR[1][2]+dL_dR[2][1]) + 2*q.w*(dL_dR[2][0]-dL_dR[0][2]);
		dL_dquat.z =   2*q.x*(dL_dR[0][2]+dL_dR[2][0]) + 2*q.y*(dL_dR[1][2]+dL_dR[2][1]) - 4*q.z*(dL_dR[0][0]+dL_dR[1][1]) + 2*q.w*(dL_dR[0][1]-dL_dR[1][0]);
	}

}

void rotScale3D_Forward(
	size_t N,
	const float* in_scales,
	const float* in_quats,
	float* out_matrices,
	bool scale_first
)
{
	rotScale3D_ForwardImpl<<<(N + UTIL_NUM_THREADS - 1) / UTIL_NUM_THREADS, UTIL_NUM_THREADS>>>(
		N,
		(const glm::vec3*)in_scales,
		(const glm::quat*)in_quats,
		(glm::mat3*)out_matrices,
		scale_first
	);
}

void rotScale3D_Backward(
	size_t N,
	const float* in_dL_dM,
	const float* saved_scales,
	const float* saved_quats,
	float* out_dL_dscales,
	float* out_dL_dquats,
	bool scale_first
)
{
	rotScale3D_BackwardImpl<<<(N + UTIL_NUM_THREADS - 1) / UTIL_NUM_THREADS, UTIL_NUM_THREADS>>>(
		N,
		(const glm::mat3*)in_dL_dM,
		(const glm::vec3*)saved_scales,
		(const glm::quat*)saved_quats,
		(glm::vec3*)out_dL_dscales,
		(glm::quat*)out_dL_dquats,
		scale_first
	);
}

}
