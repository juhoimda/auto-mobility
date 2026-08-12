/*
extract_db_rgbd.cpp — RTAB-Map DB → RGB-D + pose + intrinsics 추출기

RTAB-Map DB(SQLite)에서 최적화된 pose(Node.pose)와 원본 RGB-D(Data.image/depth)를
꺼내 TSDF 재구성용 폴더로 저장한다.

출력 구조 (outdir/):
  color/<n>.jpg     컬러 프레임 (원본 JPEG)
  depth/<n>.png     16UC1 depth (mm, 0=무효) — .rvl 디코드
  frames.txt        "seq node_id stamp color_file depth_file"
  intrinsics.txt    fx fy cx cy width height   (depth 카메라 K)
  local_transform.txt  "xyz x y z | rpy roll pitch yaw"  (camera_link→depth_optical)
  poses.txt         TUM: "stamp x y z qx qy qz qw"  (map→depth_optical)

좌표계 규약:
  Node.pose   = map → camera_link (base link) [RTAB-Map 저장값]
  localTransform = camera_link → depth optical frame
  map→optical = Node.pose * localTransform
*/

#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/Compression.h>
#include <rtabmap/core/Transform.h>
#include <sqlite3.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>

using namespace rtabmap;

static cv::Mat readBlob(sqlite3* db, sqlite3_stmt* stmt, int col)
{
    cv::Mat out;
    const unsigned char* data = static_cast<const unsigned char*>(sqlite3_column_blob(stmt, col));
    int n = sqlite3_column_bytes(stmt, col);
    if (data && n > 0)
    {
        cv::Mat m(1, n, CV_8UC1);
        memcpy(m.data, data, n);
        out = m;
    }
    return out;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        printf("usage: %s <session.db> <outdir>\n", argv[0]);
        return 1;
    }
    const std::string dbPath = argv[1];
    const std::string outDir = argv[2];

    sqlite3* db = 0;
    if (sqlite3_open(dbPath.c_str(), &db) != SQLITE_OK)
    {
        fprintf(stderr, "cannot open DB: %s\n", dbPath.c_str());
        return 1;
    }

    // ── 카메라 모델 (calibration blob) ─────────────────────────────
    CameraModel camModel;
    {
        sqlite3_stmt* stmt = 0;
        sqlite3_prepare_v2(db, "SELECT calibration FROM Data WHERE calibration IS NOT NULL LIMIT 1", -1, &stmt, 0);
        if (sqlite3_step(stmt) == SQLITE_ROW)
        {
            cv::Mat calib = readBlob(db, stmt, 0);
            if (!calib.empty())
                camModel.deserialize(calib.data, calib.cols);
        }
        sqlite3_finalize(stmt);
    }
    if (!camModel.isValidForProjection())
    {
        fprintf(stderr, "no valid camera model found in DB\n");
        sqlite3_close(db);
        return 1;
    }
    printf("depth camera: fx=%.4f fy=%.4f cx=%.4f cy=%.4f %dx%d\n",
           camModel.fx(), camModel.fy(), camModel.cx(), camModel.cy(),
           camModel.imageWidth(), camModel.imageHeight());

    // ── output dirs ────────────────────────────────────────────────
    std::string colorDir = outDir + "/color";
    std::string depthDir = outDir + "/depth";
    system(("mkdir -p " + colorDir + " " + depthDir).c_str());

    // ── frames (Data) ──────────────────────────────────────────────
    std::ofstream framesOut(outDir + "/frames.txt");
    int seq = 0;
    {
        sqlite3_stmt* stmt = 0;
        sqlite3_prepare_v2(db,
            "SELECT id, image, depth FROM Data WHERE depth IS NOT NULL ORDER BY id", -1, &stmt, 0);
        while (sqlite3_step(stmt) == SQLITE_ROW)
        {
            int id = sqlite3_column_int(stmt, 0);
            cv::Mat imgBlob = readBlob(db, stmt, 1);
            cv::Mat depthBlob = readBlob(db, stmt, 2);
            ++seq;
            char nameBuf[64];
            snprintf(nameBuf, sizeof(nameBuf), "%06d", seq);
            std::string colorFile = colorDir + "/" + nameBuf + ".jpg";
            std::string depthFile = depthDir + "/" + nameBuf + ".png";

            // depth decode (.rvl)
            cv::Mat depth;
            if (!depthBlob.empty())
            {
                std::vector<unsigned char> bytes(depthBlob.data, depthBlob.data + depthBlob.cols);
                depth = uncompressImage(bytes);
            }
            if (depth.empty() || depth.type() != CV_16UC1)
            {
                fprintf(stderr, "  skip node %d: depth decode failed\n", id);
                continue;
            }

            // color decode (JPEG)
            cv::Mat color;
            if (!imgBlob.empty())
                color = cv::imdecode(imgBlob, cv::IMREAD_UNCHANGED);

            cv::imwrite(depthFile, depth);
            if (!color.empty())
            {
                if (color.type() != CV_8UC3) cv::cvtColor(color, color, cv::COLOR_GRAY2BGR);
                cv::imwrite(colorFile, color);
            }
            else
            {
                colorFile = "none";
            }

            // pose는 rtabmap-export --poses_camera 로 별도 취득 (node id로 매칭)
            framesOut << seq << " " << id
                      << " color/" << nameBuf << ".jpg depth/" << nameBuf << ".png\n";
        }
        sqlite3_finalize(stmt);
    }
    framesOut.close();

    // ── intrinsics + local transform ───────────────────────────────
    {
        std::ofstream k(outDir + "/intrinsics.txt");
        k << camModel.fx() << " " << camModel.fy() << " " << camModel.cx() << " " << camModel.cy()
          << " " << camModel.imageWidth() << " " << camModel.imageHeight() << "\n";
    }
    {
        const Transform& lt = camModel.localTransform();
        const float* d = lt.data();
        Eigen::Matrix4d m = lt.toEigen4d();
        Eigen::Vector3d euler = m.block<3, 3>(0, 0).eulerAngles(0, 1, 2);
        std::ofstream l(outDir + "/local_transform.txt");
        l << d[3] << " " << d[7] << " " << d[11] << " " << euler.x() << " " << euler.y() << " " << euler.z() << "\n";
    }
    printf("exported %d RGB-D frames -> %s\n", seq, outDir.c_str());
    sqlite3_close(db);
    return 0;
}
