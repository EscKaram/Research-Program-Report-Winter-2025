#include <netcdf.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

struct Options {
    std::string inputPath;
    std::string outputPath;
    std::string variableName;
    std::string outputVariableName;
    std::vector<std::string> targetOrder;
    std::string chunkDimName;
    std::size_t chunkSize = 1;
};

static void checkNc(int status, const std::string &where) {
    if (status != NC_NOERR) {
        throw std::runtime_error(where + ": " + std::string(nc_strerror(status)));
    }
}

static std::string trim(const std::string &s) {
    const auto first = s.find_first_not_of(" \t");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t");
    return s.substr(first, last - first + 1);
}

static std::vector<std::string> splitList(const std::string &csv) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start < csv.size()) {
        const auto comma = csv.find(',', start);
        const auto token = trim(csv.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if (!token.empty()) parts.push_back(token);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return parts;
}

static Options parseArgs(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--input" && i + 1 < argc) {
            opt.inputPath = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            opt.outputPath = argv[++i];
        } else if (arg == "--var" && i + 1 < argc) {
            opt.variableName = argv[++i];
        } else if (arg == "--out-var" && i + 1 < argc) {
            opt.outputVariableName = argv[++i];
        } else if (arg == "--order" && i + 1 < argc) {
            opt.targetOrder = splitList(argv[++i]);
        } else if (arg == "--chunk" && i + 1 < argc) {
            opt.chunkSize = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--chunk-dim" && i + 1 < argc) {
            opt.chunkDimName = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: reshape_netcdf --input in.nc --output out.nc "
                         "--var air --order time,level,lat,lon "
                         "[--out-var air_reshaped] [--chunk 4] [--chunk-dim time]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }
    if (opt.inputPath.empty() || opt.outputPath.empty() || opt.variableName.empty() || opt.targetOrder.empty()) {
        throw std::runtime_error("Missing required arguments. Run with --help for usage.");
    }
    if (opt.outputVariableName.empty()) {
        opt.outputVariableName = opt.variableName;
    }
    if (opt.chunkSize == 0) {
        throw std::runtime_error("--chunk must be >= 1");
    }
    return opt;
}

static void copyAttributes(int srcNcid, int srcVarId, int dstNcid, int dstVarId) {
    int natt = 0;
    if (srcVarId == NC_GLOBAL) {
        checkNc(nc_inq_natts(srcNcid, &natt), "nc_inq_natts");
    } else {
        checkNc(nc_inq_varnatts(srcNcid, srcVarId, &natt), "nc_inq_varnatts");
    }
    for (int i = 0; i < natt; ++i) {
        char name[NC_MAX_NAME + 1] = {0};
        if (srcVarId == NC_GLOBAL) {
            checkNc(nc_inq_attname(srcNcid, NC_GLOBAL, i, name), "nc_inq_attname(global)");
            checkNc(nc_copy_att(srcNcid, NC_GLOBAL, name, dstNcid, NC_GLOBAL), "nc_copy_att(global)");
        } else {
            checkNc(nc_inq_attname(srcNcid, srcVarId, i, name), "nc_inq_attname(var)");
            checkNc(nc_copy_att(srcNcid, srcVarId, name, dstNcid, dstVarId), "nc_copy_att(var)");
        }
    }
}

struct CoordVarInfo {
    int srcVarId = -1;
    int dstVarId = -1;
    std::string name;
    size_t len = 0;
    nc_type type = NC_NAT;
};

static std::vector<size_t> makeStrides(const std::vector<size_t> &sizes) {
    std::vector<size_t> strides(sizes.size(), 1);
    for (int i = static_cast<int>(sizes.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * sizes[i + 1];
    }
    return strides;
}

template <typename T>
static void reshapeChunks(int inNcid,
                          int inVarId,
                          int outNcid,
                          int outVarId,
                          const std::vector<int> &perm,
                          const std::vector<size_t> &srcSizes,
                          std::size_t chunkDim,
                          std::size_t chunkSize) {
    const auto rank = srcSizes.size();
    std::vector<size_t> outSizes(rank);
    for (std::size_t i = 0; i < rank; ++i) outSizes[i] = srcSizes[perm[i]];

    std::vector<size_t> readStart(rank, 0), readCount = srcSizes;
    std::vector<size_t> writeStart(rank, 0), writeCount = outSizes;

    const size_t outer = srcSizes[chunkDim];

    for (size_t offset = 0; offset < outer; offset += chunkSize) {
        const size_t thisChunk = std::min(chunkSize, outer - offset);
        readStart[chunkDim] = offset;
        readCount[chunkDim] = thisChunk;

        for (std::size_t j = 0; j < rank; ++j) {
            if (static_cast<std::size_t>(perm[j]) == chunkDim) {
                writeStart[j] = offset;
                writeCount[j] = thisChunk;
            } else {
                writeStart[j] = 0;
                writeCount[j] = outSizes[j];
            }
        }

        size_t chunkElements = 1;
        for (auto c : readCount) chunkElements *= c;

        std::vector<T> inBuf(chunkElements);
        std::vector<T> outBuf(chunkElements);

        if constexpr (std::is_same<T, float>::value) {
            checkNc(nc_get_vara_float(inNcid, inVarId, readStart.data(), readCount.data(), inBuf.data()),
                    "nc_get_vara_float");
        } else {
            checkNc(nc_get_vara_double(inNcid, inVarId, readStart.data(), readCount.data(), inBuf.data()),
                    "nc_get_vara_double");
        }

        const auto inStrides = makeStrides(readCount);
        const auto outStrides = makeStrides(writeCount);

        std::vector<size_t> outIndex(rank, 0);
        std::vector<size_t> inIndex(rank, 0);

        for (size_t linearOut = 0; linearOut < chunkElements; ++linearOut) {
            size_t tmp = linearOut;
            for (std::size_t d = 0; d < rank; ++d) {
                outIndex[d] = tmp / outStrides[d];
                tmp -= outIndex[d] * outStrides[d];
            }
            for (std::size_t d = 0; d < rank; ++d) {
                inIndex[perm[d]] = outIndex[d];
            }
            size_t linearIn = 0;
            for (std::size_t d = 0; d < rank; ++d) {
                linearIn += inIndex[d] * inStrides[d];
            }
            outBuf[linearOut] = inBuf[linearIn];
        }

        if constexpr (std::is_same<T, float>::value) {
            checkNc(nc_put_vara_float(outNcid, outVarId, writeStart.data(), writeCount.data(), outBuf.data()),
                    "nc_put_vara_float");
        } else {
            checkNc(nc_put_vara_double(outNcid, outVarId, writeStart.data(), writeCount.data(), outBuf.data()),
                    "nc_put_vara_double");
        }
    }
}

static void copyCoordinateData(int inNcid, const CoordVarInfo &info, int outNcid) {
    if (info.srcVarId < 0 || info.dstVarId < 0) return;
    if (info.type == NC_DOUBLE) {
        std::vector<double> buf(info.len);
        checkNc(nc_get_var_double(inNcid, info.srcVarId, buf.data()), "nc_get_var_double(coord)");
        checkNc(nc_put_var_double(outNcid, info.dstVarId, buf.data()), "nc_put_var_double(coord)");
    } else if (info.type == NC_FLOAT) {
        std::vector<float> buf(info.len);
        checkNc(nc_get_var_float(inNcid, info.srcVarId, buf.data()), "nc_get_var_float(coord)");
        checkNc(nc_put_var_float(outNcid, info.dstVarId, buf.data()), "nc_put_var_float(coord)");
    } else if (info.type == NC_INT || info.type == NC_UINT) {
        std::vector<int> buf(info.len);
        checkNc(nc_get_var_int(inNcid, info.srcVarId, buf.data()), "nc_get_var_int(coord)");
        checkNc(nc_put_var_int(outNcid, info.dstVarId, buf.data()), "nc_put_var_int(coord)");
    } else {
        throw std::runtime_error("Unsupported coordinate variable type for " + info.name);
    }
}

int main(int argc, char **argv) {
    try {
        const auto opt = parseArgs(argc, argv);

        int inNcid = -1;
        checkNc(nc_open(opt.inputPath.c_str(), NC_NOWRITE, &inNcid), "nc_open");

        int inVarId = -1;
        if (nc_inq_varid(inNcid, opt.variableName.c_str(), &inVarId) != NC_NOERR) {
            throw std::runtime_error("Variable not found: " + opt.variableName);
        }

        char varName[NC_MAX_NAME + 1] = {0};
        nc_type varType = NC_NAT;
        int rank = 0;
        int dimIds[NC_MAX_VAR_DIMS] = {0};
        int natts = 0;
        checkNc(nc_inq_var(inNcid, inVarId, varName, &varType, &rank, dimIds, &natts), "nc_inq_var");
        if (rank <= 0) throw std::runtime_error("Variable has no dimensions.");

        if (static_cast<int>(opt.targetOrder.size()) != rank) {
            throw std::runtime_error("Dimension count mismatch between --order and variable.");
        }

        std::vector<std::string> dimNames(rank);
        std::vector<size_t> dimSizes(rank);
        std::vector<int> dimIdVec(rank);
        for (int i = 0; i < rank; ++i) dimIdVec[i] = dimIds[i];

        int nunlim = 0;
        int unlimIds[NC_MAX_DIMS] = {0};
        checkNc(nc_inq_unlimdims(inNcid, &nunlim, unlimIds), "nc_inq_unlimdims");
        std::vector<int> unlimList(unlimIds, unlimIds + nunlim);

        auto isUnlimited = [&](int dimId) {
            return std::find(unlimList.begin(), unlimList.end(), dimId) != unlimList.end();
        };

        for (int i = 0; i < rank; ++i) {
            char name[NC_MAX_NAME + 1] = {0};
            size_t len = 0;
            checkNc(nc_inq_dim(inNcid, dimIds[i], name, &len), "nc_inq_dim");
            dimNames[i] = name;
            dimSizes[i] = len;
        }

        std::vector<int> perm(rank, -1);
        for (std::size_t i = 0; i < opt.targetOrder.size(); ++i) {
            const auto &name = opt.targetOrder[i];
            const auto it = std::find(dimNames.begin(), dimNames.end(), name);
            if (it == dimNames.end()) {
                throw std::runtime_error("Dimension " + name + " not found in variable.");
            }
            int idx = static_cast<int>(std::distance(dimNames.begin(), it));
            if (std::find(perm.begin(), perm.end(), idx) != perm.end()) {
                throw std::runtime_error("Duplicate dimension in --order: " + name);
            }
            perm[i] = idx;
        }

        std::size_t chunkDim = 0;
        if (!opt.chunkDimName.empty()) {
            const auto it = std::find(dimNames.begin(), dimNames.end(), opt.chunkDimName);
            if (it == dimNames.end()) throw std::runtime_error("Chunk dimension not found: " + opt.chunkDimName);
            chunkDim = static_cast<std::size_t>(std::distance(dimNames.begin(), it));
        }

        int outNcid = -1;
        checkNc(nc_create(opt.outputPath.c_str(), NC_NETCDF4, &outNcid), "nc_create");

        std::vector<int> outDimIds(rank, -1);
        std::map<std::string, int> outDimIdByName;
        for (int i = 0; i < rank; ++i) {
            int srcIdx = perm[i];
            size_t len = dimSizes[srcIdx];
            int srcDimId = dimIds[srcIdx];
            int outDimId = -1;
            if (isUnlimited(srcDimId)) {
                checkNc(nc_def_dim(outNcid, dimNames[srcIdx].c_str(), NC_UNLIMITED, &outDimId), "nc_def_dim(unlim)");
            } else {
                checkNc(nc_def_dim(outNcid, dimNames[srcIdx].c_str(), len, &outDimId), "nc_def_dim");
            }
            outDimIds[i] = outDimId;
            outDimIdByName[dimNames[srcIdx]] = outDimId;
        }

        std::vector<CoordVarInfo> coordVars;
        for (int i = 0; i < rank; ++i) {
            const auto &dimName = dimNames[i];
            int coordVarId = -1;
            if (nc_inq_varid(inNcid, dimName.c_str(), &coordVarId) != NC_NOERR) continue;
            int coordRank = 0;
            int coordDimIds[NC_MAX_VAR_DIMS] = {0};
            nc_type coordType = NC_NAT;
            int coordNatts = 0;
            checkNc(nc_inq_var(inNcid, coordVarId, nullptr, &coordType, &coordRank, coordDimIds, &coordNatts),
                    "nc_inq_var(coord)");
            if (coordRank != 1 || coordDimIds[0] != dimIds[i]) continue;

            int outDimId = outDimIdByName[dimName];
            int outVarId = -1;
            checkNc(nc_def_var(outNcid, dimName.c_str(), coordType, 1, &outDimId, &outVarId), "nc_def_var(coord)");

            CoordVarInfo info;
            info.srcVarId = coordVarId;
            info.dstVarId = outVarId;
            info.name = dimName;
            info.len = dimSizes[i];
            info.type = coordType;
            coordVars.push_back(info);
        }

        std::vector<int> outVarDimIds(rank);
        for (int i = 0; i < rank; ++i) outVarDimIds[i] = outDimIds[i];
        int outVarId = -1;
        checkNc(nc_def_var(outNcid, opt.outputVariableName.c_str(), varType, rank, outVarDimIds.data(), &outVarId),
                "nc_def_var(main)");

        checkNc(nc_enddef(outNcid), "nc_enddef");

        copyAttributes(inNcid, NC_GLOBAL, outNcid, NC_GLOBAL);
        for (const auto &cv : coordVars) {
            copyAttributes(inNcid, cv.srcVarId, outNcid, cv.dstVarId);
        }
        copyAttributes(inNcid, inVarId, outNcid, outVarId);

        for (const auto &cv : coordVars) {
            copyCoordinateData(inNcid, cv, outNcid);
        }

        if (varType == NC_FLOAT) {
            reshapeChunks<float>(inNcid, inVarId, outNcid, outVarId, perm, dimSizes, chunkDim, opt.chunkSize);
        } else if (varType == NC_DOUBLE) {
            reshapeChunks<double>(inNcid, inVarId, outNcid, outVarId, perm, dimSizes, chunkDim, opt.chunkSize);
        } else {
            throw std::runtime_error("Unsupported variable type; only float/double are handled.");
        }

        nc_close(inNcid);
        nc_close(outNcid);
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
