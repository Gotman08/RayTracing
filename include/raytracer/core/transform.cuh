#ifndef RAYTRACER_CORE_TRANSFORM_CUH
#define RAYTRACER_CORE_TRANSFORM_CUH

#include "raytracer/core/vec3.cuh"
#include "raytracer/core/cuda_utils.cuh"

namespace rt {

class Matrix4 {
public:
    float m[4][4];

    __host__ __device__ Matrix4() {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                m[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    __host__ __device__ static Matrix4 identity() {
        return Matrix4();
    }

    __host__ __device__ static Matrix4 translate(const Vec3& t) {
        Matrix4 mat;
        mat.m[0][3] = t.x;
        mat.m[1][3] = t.y;
        mat.m[2][3] = t.z;
        return mat;
    }

    __host__ __device__ static Matrix4 scale(const Vec3& s) {
        Matrix4 mat;
        mat.m[0][0] = s.x;
        mat.m[1][1] = s.y;
        mat.m[2][2] = s.z;
        return mat;
    }

    __host__ __device__ static Matrix4 rotate_x(float angle) {
        Matrix4 mat;
        float c = cosf(degrees_to_radians(angle));
        float s = sinf(degrees_to_radians(angle));
        mat.m[1][1] = c;  mat.m[1][2] = -s;
        mat.m[2][1] = s;  mat.m[2][2] = c;
        return mat;
    }

    __host__ __device__ static Matrix4 rotate_y(float angle) {
        Matrix4 mat;
        float c = cosf(degrees_to_radians(angle));
        float s = sinf(degrees_to_radians(angle));
        mat.m[0][0] = c;  mat.m[0][2] = s;
        mat.m[2][0] = -s; mat.m[2][2] = c;
        return mat;
    }

    __host__ __device__ static Matrix4 rotate_z(float angle) {
        Matrix4 mat;
        float c = cosf(degrees_to_radians(angle));
        float s = sinf(degrees_to_radians(angle));
        mat.m[0][0] = c;  mat.m[0][1] = -s;
        mat.m[1][0] = s;  mat.m[1][1] = c;
        return mat;
    }

    __host__ __device__ Matrix4 operator*(const Matrix4& other) const {
        Matrix4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) {
                    result.m[i][j] += m[i][k] * other.m[k][j];
                }
            }
        }
        return result;
    }

    __host__ __device__ Vec3 transform_point(const Vec3& p) const {
        float x = m[0][0]*p.x + m[0][1]*p.y + m[0][2]*p.z + m[0][3];
        float y = m[1][0]*p.x + m[1][1]*p.y + m[1][2]*p.z + m[1][3];
        float z = m[2][0]*p.x + m[2][1]*p.y + m[2][2]*p.z + m[2][3];
        float w = m[3][0]*p.x + m[3][1]*p.y + m[3][2]*p.z + m[3][3];
        if (w != 1.0f && w != 0.0f) {
            x /= w; y /= w; z /= w;
        }
        return Vec3(x, y, z);
    }

    __host__ __device__ Vec3 transform_vector(const Vec3& v) const {
        return Vec3(
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z
        );
    }

    __host__ __device__ Vec3 transform_normal(const Vec3& n) const {
        // For normals, use inverse transpose (simplified for orthogonal matrices)
        return transform_vector(n).normalized();
    }
};

} // namespace rt

#endif // RAYTRACER_CORE_TRANSFORM_CUH
