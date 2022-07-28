package(default_visibility = ["//visibility:public"])

cc_library(
    name = "merlinkv",
    hdrs = [
        "cpp/include/merlin_hashtable.cuh"
    ] + glob(["cpp/include/merlin/*.cuh"]),
)

cc_library(
    name = "merlinkv-test",
    hdrs = [
        "cpp/include/merlin_hashtable.cuh"
    ] + glob(["cpp/include/merlin/*.cuh"]),
    srcs = [
        "tests/merlin_hashtable_test.cc.cu",
    ]
)
