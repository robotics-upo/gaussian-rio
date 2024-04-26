#include "common.h"

namespace {

void checkTensor(torch::Tensor t, const char* name, torch::Dtype type = torch::kFloat32)
{
	TORCH_CHECK(t.is_contiguous(), "Input '", name, "' must be contiguous");
	TORCH_CHECK(t.options().dtype().isScalarType(type), "Input '", name, "' must be ", torch::toString(type));
	TORCH_CHECK(t.options().device().is_cuda(), "Input '", name, "' must be CUDA");
}

torch::Tensor rotScale3D_Forward(torch::Tensor scales, torch::Tensor quats, bool scale_first)
{
	checkTensor(scales, "scales");
	checkTensor(quats, "quats");

	TORCH_CHECK(scales.ndimension() == 2 && scales.size(1) == 3, "Input 'scales' must be a tensor of shape (N,3)");
	TORCH_CHECK(quats.ndimension() == 2 && quats.size(1) == 4, "Input 'quats' must be a tensor of shape (N,4)");

	int64_t N = scales.size(0);
	TORCH_CHECK(quats.size(0) == N, "Inputs 'scales' and 'quats' must have the same batch size");

	auto options = torch::TensorOptions(torch::kFloat32).device(torch::Device(torch::kCUDA));
	auto out = torch::empty({ N, 3, 3 }, options);

	gaussian_utils::rotScale3D_Forward(N,
		scales.mutable_data_ptr<float>(),
		quats.mutable_data_ptr<float>(),
		out.mutable_data_ptr<float>(),
		scale_first
	);

	return out;
}

std::tuple<torch::Tensor, torch::Tensor> rotScale3D_Backward(torch::Tensor grad, torch::Tensor scales, torch::Tensor quats, bool scale_first)
{
	checkTensor(grad, "grad");
	checkTensor(scales, "scales");
	checkTensor(quats, "quats");

	TORCH_CHECK(grad.ndimension() == 3 && grad.size(1) == 3 && grad.size(2) == 3, "Input 'grad' must be a tensor of shape (N,3,3)");
	TORCH_CHECK(scales.ndimension() == 2 && scales.size(1) == 3, "Input 'scales' must be a tensor of shape (N,3)");
	TORCH_CHECK(quats.ndimension() == 2 && quats.size(1) == 4, "Input 'quats' must be a tensor of shape (N,4)");

	int64_t N = grad.size(0);
	TORCH_CHECK(scales.size(0) == N, "Inputs 'grad' and 'scales' must have the same batch size");
	TORCH_CHECK(quats.size(0) == N, "Inputs 'grad' and 'quats' must have the same batch size");

	auto options = torch::TensorOptions(torch::kFloat32).device(torch::Device(torch::kCUDA));
	auto out_dL_dscales = torch::empty({ N, 3 }, options);
	auto out_dL_dquats  = torch::empty({ N, 4 }, options);

	gaussian_utils::rotScale3D_Backward(N,
		grad.mutable_data_ptr<float>(),
		scales.mutable_data_ptr<float>(),
		quats.mutable_data_ptr<float>(),
		out_dL_dscales.mutable_data_ptr<float>(),
		out_dL_dquats.mutable_data_ptr<float>(),
		scale_first
	);

	return { out_dL_dscales, out_dL_dquats };
}

std::tuple<torch::Tensor, torch::Tensor> nearestCenter3D_Forward(torch::Tensor points, torch::Tensor centers)
{
	checkTensor(points, "points");
	checkTensor(centers, "centers");

	TORCH_CHECK(points.ndimension() == 2 && points.size(1) == 3, "Input 'points' must be a tensor of shape (N,3)");
	TORCH_CHECK(centers.ndimension() == 2 && centers.size(1) == 3, "Input 'centers' must be a tensor of shape (N,3)");

	int64_t N_points = points.size(0);
	int64_t N_centers = centers.size(0);

	auto out_vectors = torch::empty({ N_points, 3 },
		torch::TensorOptions(torch::kFloat32).device(torch::Device(torch::kCUDA))
	);

	auto out_indices = torch::empty({ N_points },
		torch::TensorOptions(torch::kInt64).device(torch::Device(torch::kCUDA))
	);

	gaussian_utils::nearestCenter3D_Forward(N_points, N_centers,
		points.mutable_data_ptr<float>(),
		centers.mutable_data_ptr<float>(),
		out_vectors.mutable_data_ptr<float>(),
		(size_t*)out_indices.mutable_data_ptr<long>()
	);

	return { out_vectors, out_indices };
}

std::tuple<torch::Tensor, torch::Tensor> nearestCenter3D_Backward(int64_t N_centers, torch::Tensor grad, torch::Tensor indices)
{
	TORCH_CHECK(N_centers > 0, "Input 'N_centers' must be positive");
	checkTensor(grad, "grad");
	checkTensor(indices, "indices", torch::kInt64);

	TORCH_CHECK(grad.ndimension() == 2 && grad.size(1) == 3, "Input 'grad' must be a tensor of shape (N,3)");

	int64_t N_points = grad.size(0);
	TORCH_CHECK(indices.ndimension() == 1 && indices.size(0) == N_points, "Input 'indices' must be a tensor of shape (N)");

	auto options = torch::TensorOptions(torch::kFloat32).device(torch::Device(torch::kCUDA));

	auto out_dL_dpoints = torch::zeros({ N_points, 3 }, options);
	auto out_dL_dcenters = torch::zeros({ N_centers, 3 }, options);

	gaussian_utils::nearestCenter3D_Backward(N_points, N_centers,
		grad.mutable_data_ptr<float>(),
		(size_t*)indices.mutable_data_ptr<long>(),
		out_dL_dpoints.mutable_data_ptr<float>(),
		out_dL_dcenters.mutable_data_ptr<float>()
	);

	return { out_dL_dpoints, out_dL_dcenters };
}

std::tuple<torch::Tensor, torch::Tensor> nearestGaussian3D_Forward(torch::Tensor points, torch::Tensor centers, torch::Tensor matrices)
{
	checkTensor(points, "points");
	checkTensor(centers, "centers");
	checkTensor(matrices, "matrices");

	int64_t N_points = points.size(0);
	int64_t N_gaussians = centers.size(0);

	auto out_sqmahal = torch::empty({ N_points },
		torch::TensorOptions(torch::kFloat32).device(torch::Device(torch::kCUDA))
	);

	auto out_indices = torch::empty({ N_points },
		torch::TensorOptions(torch::kInt64).device(torch::Device(torch::kCUDA))
	);

	gaussian_utils::nearestGaussian3D_Forward(N_points, N_gaussians,
		points.mutable_data_ptr<float>(),
		centers.mutable_data_ptr<float>(),
		matrices.mutable_data_ptr<float>(),
		out_sqmahal.mutable_data_ptr<float>(),
		(size_t*)out_indices.mutable_data_ptr<long>()
	);

	return { out_sqmahal, out_indices };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> nearestGaussian3D_Backward(torch::Tensor grad, torch::Tensor points, torch::Tensor centers, torch::Tensor matrices, torch::Tensor indices)
{
	checkTensor(grad, "grad");
	checkTensor(points, "points");
	checkTensor(centers, "centers");
	checkTensor(matrices, "matrices");
	checkTensor(indices, "indices", torch::kInt64);

	int64_t N_points = grad.size(0);
	int64_t N_gaussians = centers.size(0);

	auto options = torch::TensorOptions(torch::kFloat32).device(torch::Device(torch::kCUDA));

	auto out_dL_dpoints = torch::empty({ N_points, 3 }, options);
	auto out_dL_dcenters = torch::zeros({ N_gaussians, 3 }, options);
	auto out_dL_dmatrices = torch::zeros({ N_gaussians, 3, 3 }, options);

	gaussian_utils::nearestGaussian3D_Backward(N_points, N_gaussians,
		grad.mutable_data_ptr<float>(),
		points.mutable_data_ptr<float>(),
		centers.mutable_data_ptr<float>(),
		matrices.mutable_data_ptr<float>(),
		(size_t*)indices.mutable_data_ptr<long>(),
		out_dL_dpoints.mutable_data_ptr<float>(),
		out_dL_dcenters.mutable_data_ptr<float>(),
		out_dL_dmatrices.mutable_data_ptr<float>()
	);

	return { out_dL_dpoints, out_dL_dcenters, out_dL_dmatrices };
}

torch::Tensor nearestGaussian3D_BackwardLite(torch::Tensor grad, torch::Tensor points, torch::Tensor centers, torch::Tensor matrices, torch::Tensor indices)
{
	checkTensor(grad, "grad");
	checkTensor(points, "points");
	checkTensor(centers, "centers");
	checkTensor(matrices, "matrices");
	checkTensor(indices, "indices", torch::kInt64);

	int64_t N_points = grad.size(0);
	int64_t N_gaussians = centers.size(0);

	auto out_dL_dpoints = torch::empty({ N_points, 3 },
		torch::TensorOptions(torch::kFloat32).device(torch::Device(torch::kCUDA))
	);

	gaussian_utils::nearestGaussian3D_Backward(N_points, N_gaussians,
		grad.mutable_data_ptr<float>(),
		points.mutable_data_ptr<float>(),
		centers.mutable_data_ptr<float>(),
		matrices.mutable_data_ptr<float>(),
		(size_t*)indices.mutable_data_ptr<long>(),
		out_dL_dpoints.mutable_data_ptr<float>()
	);

	return out_dL_dpoints;
}

torch::Tensor indexedTransform3D_Forward(torch::Tensor points, torch::Tensor matrices, torch::Tensor indices)
{
	checkTensor(points, "points");
	checkTensor(matrices, "matrices");
	checkTensor(indices, "indices", torch::kInt64);

	TORCH_CHECK(points.ndimension() == 2 && points.size(1) == 3, "Input 'points' must be a tensor of shape (N,3)");
	TORCH_CHECK(matrices.ndimension() == 3 && matrices.size(1) == 3 && matrices.size(2) == 3, "Input 'matrices' must be a tensor of shape (M,3,3)");
	TORCH_CHECK(indices.ndimension() == 1, "Input 'indices' must be a tensor of shape (N)");

	int64_t N_points = points.size(0);
	TORCH_CHECK(indices.size(0) == N_points, "Input 'indices' must contain the same number of elements as 'points'");

	auto out_points = torch::zeros({ N_points, 3 },
		torch::TensorOptions(torch::kFloat32).device(torch::Device(torch::kCUDA))
	);

	gaussian_utils::indexedTransform3D_Forward(N_points,
		points.mutable_data_ptr<float>(),
		matrices.mutable_data_ptr<float>(),
		(size_t*)indices.mutable_data_ptr<long>(),
		out_points.mutable_data_ptr<float>()
	);

	return out_points;
}

std::tuple<torch::Tensor, torch::Tensor> indexedTransform3D_Backward(torch::Tensor grad, torch::Tensor points, torch::Tensor matrices, torch::Tensor indices)
{
	// xx safety

	int64_t N_points = grad.size(0);
	int64_t N_gaussians = matrices.size(0);

	auto options = torch::TensorOptions(torch::kFloat32).device(torch::Device(torch::kCUDA));
	auto out_dL_dpoints   = torch::empty({ N_points,    3    }, options);
	auto out_dL_dmatrices = torch::zeros({ N_gaussians, 3, 3 }, options);

	gaussian_utils::indexedTransform3D_Backward(N_points,
		grad.mutable_data_ptr<float>(),
		points.mutable_data_ptr<float>(),
		matrices.mutable_data_ptr<float>(),
		(size_t*)indices.mutable_data_ptr<long>(),
		out_dL_dpoints.mutable_data_ptr<float>(),
		out_dL_dmatrices.mutable_data_ptr<float>()
	);

	return { out_dL_dpoints, out_dL_dmatrices };
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("rot_scale_3d_forward", rotScale3D_Forward, "Calculates 3D rotation/scale matrix from scale/quaternion");
	m.def("rot_scale_3d_backward", rotScale3D_Backward, "Gradient of rot_scale_3d_forward");

	m.def("nearest_center_3d_forward", nearestCenter3D_Forward, "Calculates 3D nearest center");
	m.def("nearest_center_3d_backward", nearestCenter3D_Backward, "Gradient of nearest_center_3d_forward");

	m.def("nearest_gaussian_3d_forward", nearestGaussian3D_Forward, "Calculates 3D nearest gaussian");
	m.def("nearest_gaussian_3d_backward", nearestGaussian3D_Backward, "Gradient of nearest_gaussian_3d_forward");
	m.def("nearest_gaussian_3d_backward_lite", nearestGaussian3D_BackwardLite, "Gradient of nearest_gaussian_3d_forward (point-only version)");

	m.def("indexed_transform_3d_forward", indexedTransform3D_Forward, "Transforms 3D points according to an indexed matrix");
	m.def("indexed_transform_3d_backward", indexedTransform3D_Backward, "Gradient of indexed_transform_3d_forward");
}
