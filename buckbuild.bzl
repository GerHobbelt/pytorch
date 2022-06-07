load("//tools/build_defs:expect.bzl", "expect")

# NOTE: This file is shared by internal and OSS BUCK build.
# These load paths point to different files in internal and OSS environment
load("//tools/build_defs:fb_native_wrapper.bzl", "fb_native")
load("//tools/build_defs:fb_xplat_cxx_library.bzl", "fb_xplat_cxx_library")
load("//tools/build_defs:fb_xplat_genrule.bzl", "fb_xplat_genrule")
load("//tools/build_defs:glob_defs.bzl", "subdir_glob")
load("//tools/build_defs:type_defs.bzl", "is_list", "is_string")
load(
    ":build_variables.bzl",
    "jit_core_headers",
)
load(
    ":ufunc_defs.bzl",
    "aten_ufunc_generated_cpu_kernel_sources",
    "aten_ufunc_generated_cpu_sources",
    "aten_ufunc_generated_cuda_sources",
)

def read_bool(section, field, default):
    # @lint-ignore BUCKRESTRICTEDSYNTAX
    value = read_config(section, field)
    if value == None:
        return default
    expect(
        value == "0" or value == "1",
        "{}.{} == \"{}\", wanted \"0\" or \"1\".".format(section, field, value),
    )
    return bool(int(value))

def is_oss_build():
    return read_bool("pt", "is_oss", False)

# for targets in caffe2 root path
ROOT_NAME = "//" if is_oss_build() else "//xplat/caffe2"

# for targets in subfolders
ROOT_PATH = "//" if is_oss_build() else "//xplat/caffe2/"

# common buck configs
def get_static_dispatch_backend():
    static_dispatch_backend = native.read_config("pt", "static_dispatch_backend", None)
    if static_dispatch_backend == None:
        return []
    return static_dispatch_backend.split(";")

def get_build_from_deps_query():
    return read_bool("pt", "build_from_deps_query", True)

def get_enable_lightweight_dispatch():
    return read_bool("pt", "enable_lightweight_dispatch", False)

# common consntants
ATEN_COMPILER_FLAGS = [
    "-fexceptions",
    "-frtti",
    "-fPIC",
    "-Os",
    "-Wno-absolute-value",
    "-Wno-deprecated-declarations",
    "-Wno-macro-redefined",
    "-Wno-tautological-constant-out-of-range-compare",
    "-Wno-unknown-pragmas",
    "-Wno-unknown-warning-option",
    "-Wno-unused-function",
    "-Wno-unused-variable",
    "-Wno-pass-failed",
    "-Wno-shadow",
]

PT_COMPILER_FLAGS = [
    "-frtti",
    "-Os",
    "-Wno-unknown-pragmas",
    "-Wno-write-strings",
    "-Wno-unused-variable",
    "-Wno-unused-function",
    "-Wno-deprecated-declarations",
    "-Wno-shadow",
    "-Wno-global-constructors",
    "-Wno-missing-prototypes",
]

PT_COMPILER_FLAGS_DEFAULT = PT_COMPILER_FLAGS + [
    "-std=gnu++17",  #to accomodate for eigen
]

ATEN_PREPROCESSOR_FLAGS = [
    "-DC10_MOBILE",
    "-DCPU_CAPABILITY_DEFAULT",
    "-DCPU_CAPABILITY=DEFAULT",
    "-DCAFFE2_USE_LITE_PROTO",
    "-DATEN_CUDNN_ENABLED_FBXPLAT=0",
    "-DATEN_MKLDNN_ENABLED_FBXPLAT=0",
    "-DATEN_NNPACK_ENABLED_FBXPLAT=0",
    "-DATEN_MKL_ENABLED_FBXPLAT=0",
    "-DATEN_MKL_SEQUENTIAL_FBXPLAT=0",
    "-DUSE_PYTORCH_METAL",
    "-DUSE_PYTORCH_QNNPACK",
    "-DUSE_XNNPACK",
    "-DNO_EXPORT",
    "-DPYTORCH_QNNPACK_RUNTIME_QUANTIZATION",
    "-DAT_PARALLEL_OPENMP_FBXPLAT=0",
    "-DAT_PARALLEL_NATIVE_FBXPLAT=1",
    "-DAT_PARALLEL_NATIVE_TBB_FBXPLAT=0",
    "-DUSE_LAPACK_FBXPLAT=0",
    "-DAT_BLAS_F2C_FBXPLAT=0",
    "-DAT_BLAS_USE_CBLAS_DOT_FBXPLAT=0",
    "-DUSE_RUY_QMATMUL",
]

PT_PREPROCESSOR_FLAGS = [
    "-D_THP_CORE",
    "-DC10_MOBILE",
    "-DUSE_SCALARS",
    "-DNO_CUDNN_DESTROY_HANDLE",
    "-DNO_EXPORT",
    "-DBUILD_CAFFE2",
]

USED_PT_BACKENDS = [
    "CPU",
    "QuantizedCPU",
    "SparseCPU",  # brings ~20 kb size regression
]

# This needs to be kept in sync with https://github.com/pytorch/pytorch/blob/release/1.9/torchgen/gen.py#L892
PT_BACKEND_HEADERS = [
    "CPU",
    "CUDA",
    "CompositeExplicitAutograd",
    "CompositeImplicitAutograd",
    "Meta",
]

def get_aten_static_dispatch_backend_headers(existing_headers):
    static_backends = get_static_dispatch_backend()
    for backend in static_backends:
        if backend != "CPU":
            existing_headers["{}Functions.h".format(backend)] = ":gen_aten[{}Functions.h]".format(backend)
            existing_headers["{}Functions_inl.h".format(backend)] = ":gen_aten[{}Functions_inl.h]".format(backend)
    return existing_headers

def get_aten_codegen_extra_params(backends):
    if get_build_from_deps_query():
        extra_params = {
            "force_schema_registration": True,
        }
        static_backends = get_static_dispatch_backend()
        if static_backends:
            extra_params["static_dispatch_backend"] = static_backends
            extra_params["enabled_backends"] = static_backends
        else:
            extra_params["enabled_backends"] = backends
        return extra_params
    else:
        return {}

def gen_aten_files(
        name,
        extra_flags = {},
        visibility = [],
        compatible_with = [],
        apple_sdks = None):
    extra_params = []
    force_schema_registration = extra_flags.get("force_schema_registration", False)
    op_registration_allowlist = extra_flags.get("op_registration_allowlist", None)
    op_selection_yaml_path = extra_flags.get("op_selection_yaml_path", None)
    enabled_backends = extra_flags.get("enabled_backends", None)
    static_dispatch_backend = extra_flags.get("static_dispatch_backend", None)

    if force_schema_registration:
        extra_params.append("--force_schema_registration")
    if op_registration_allowlist != None and is_string(op_registration_allowlist):
        extra_params.append("--op_registration_whitelist")
        extra_params.append(op_registration_allowlist)
    if op_selection_yaml_path != None and is_string(op_selection_yaml_path):
        extra_params.append("--op_selection_yaml_path")
        extra_params.append(op_selection_yaml_path)
    if enabled_backends != None and is_list(enabled_backends):
        extra_params.append("--backend_whitelist")
        extra_params.extend(enabled_backends)
    if get_enable_lightweight_dispatch():
        extra_params.append("--skip_dispatcher_op_registration")
    if static_dispatch_backend:
        extra_params.append("--static_dispatch_backend")
        extra_params.extend(static_dispatch_backend)
        backends = static_dispatch_backend
    else:
        backends = enabled_backends
    fb_xplat_genrule(
        name = name,
        default_outs = ["."],
        outs = get_aten_generated_files(backends),
        cmd = "$(exe {}:gen_aten_bin) ".format(ROOT_NAME) + " ".join([
            "--source-path $(location {}:aten_src_path)/aten/src/ATen".format(ROOT_NAME),
            "--install_dir $OUT",
        ] + extra_params),
        visibility = visibility,
        compatible_with = compatible_with,
        apple_sdks = apple_sdks,
    )

def get_aten_derived_type_srcs(enabled_backends):
    return [
        "Register" + derived_type + ".cpp"
        for derived_type in enabled_backends
    ] + [
        derived_type + "Functions.h"
        for derived_type in enabled_backends
        if derived_type in PT_BACKEND_HEADERS or derived_type in get_static_dispatch_backend()
    ] + [
        derived_type + "Functions_inl.h"
        for derived_type in enabled_backends
        if derived_type in PT_BACKEND_HEADERS or derived_type in get_static_dispatch_backend()
    ]

def get_aten_generated_files(enabled_backends):
    # NB: RegisterMeta counts as an optionally enabled backend,
    # and is intentionally omitted from here
    src_files = [
        "RegisterBackendSelect.cpp",
        "RegisterCompositeImplicitAutograd.cpp",
        "RegisterCompositeExplicitAutograd.cpp",
        "CompositeViewCopyKernels.cpp",
        "RegisterSchema.cpp",
        "Declarations.yaml",
        "Functions.cpp",
        "Functions.h",
        "RedispatchFunctions.h",
        "NativeFunctions.h",
        "NativeMetaFunctions.h",
        "MethodOperators.h",
        "FunctionalInverses.h",
        "Operators.h",
        "Operators_0.cpp",
        "Operators_1.cpp",
        "Operators_2.cpp",
        "Operators_3.cpp",
        "Operators_4.cpp",
        "CompositeImplicitAutogradFunctions.h",
        "CompositeImplicitAutogradFunctions_inl.h",
        "CompositeExplicitAutogradFunctions.h",
        "CompositeExplicitAutogradFunctions_inl.h",
        "core/ATenOpList.cpp",
        "core/TensorBody.h",
        "core/TensorMethods.cpp",
        "core/aten_interned_strings.h",
    ] + get_aten_derived_type_srcs(enabled_backends)

    # This is tiresome.  A better strategy would be to unconditionally
    # generate these files, and then only actually COMPILE them depended
    # on the generated set.  C'est la vie...
    if "CPU" in enabled_backends:
        src_files.extend(aten_ufunc_generated_cpu_sources())
        src_files.extend(aten_ufunc_generated_cpu_kernel_sources())
    if "CUDA" in enabled_backends:
        # Cannot unconditionally include this, because in the Edge selective
        # build CUDA is not enabled and thus the ufunc codegen for CUDA gets
        # skipped
        src_files.extend(aten_ufunc_generated_cuda_sources())

    res = {}
    for file_name in src_files:
        res[file_name] = [file_name]
    return res

# these targets are shared by internal and OSS BUCK
def define_buck_targets(
        feature = None,
        labels = []):
    fb_xplat_cxx_library(
        name = "th_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            # TH
            ("aten/src", "TH/*.h"),
            ("aten/src", "TH/*.hpp"),
            ("aten/src", "TH/generic/*.h"),
            ("aten/src", "TH/generic/*.hpp"),
            ("aten/src", "TH/generic/simd/*.h"),
            ("aten/src", "TH/vector/*.h"),
            ("aten/src", "TH/generic/*.c"),
            ("aten/src", "TH/generic/*.cpp"),
            ("aten/src/TH", "*.h"),  # for #include <THGenerateFloatTypes.h>
            # THNN
            ("aten/src", "THNN/*.h"),
            ("aten/src", "THNN/generic/*.h"),
            ("aten/src", "THNN/generic/*.c"),
        ]),
        feature = feature,
        labels = labels,
    )

    fb_xplat_cxx_library(
        name = "aten_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            # ATen Core
            ("aten/src", "ATen/core/**/*.h"),
            ("aten/src", "ATen/ops/*.h"),
            # ATen Base
            ("aten/src", "ATen/*.h"),
            ("aten/src", "ATen/cpu/**/*.h"),
            ("aten/src", "ATen/detail/*.h"),
            ("aten/src", "ATen/quantized/*.h"),
            ("aten/src", "ATen/vulkan/*.h"),
            ("aten/src", "ATen/metal/*.h"),
            ("aten/src", "ATen/nnapi/*.h"),
            # ATen Native
            ("aten/src", "ATen/native/*.h"),
            ("aten/src", "ATen/native/ao_sparse/quantized/cpu/*.h"),
            ("aten/src", "ATen/native/cpu/**/*.h"),
            ("aten/src", "ATen/native/sparse/*.h"),
            ("aten/src", "ATen/native/nested/*.h"),
            ("aten/src", "ATen/native/quantized/*.h"),
            ("aten/src", "ATen/native/quantized/cpu/*.h"),
            ("aten/src", "ATen/native/transformers/*.h"),
            ("aten/src", "ATen/native/ufunc/*.h"),
            ("aten/src", "ATen/native/utils/*.h"),
            ("aten/src", "ATen/native/vulkan/ops/*.h"),
            ("aten/src", "ATen/native/xnnpack/*.h"),
            ("aten/src", "ATen/mps/*.h"),
            ("aten/src", "ATen/native/mps/*.h"),
            # Remove the following after modifying codegen for mobile.
            ("aten/src", "ATen/mkl/*.h"),
            ("aten/src", "ATen/native/mkl/*.h"),
            ("aten/src", "ATen/native/mkldnn/*.h"),
        ]),
        visibility = ["PUBLIC"],
        feature = feature,
        labels = labels,
    )

    fb_xplat_cxx_library(
        name = "aten_vulkan_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            ("aten/src", "ATen/native/vulkan/*.h"),
            ("aten/src", "ATen/native/vulkan/api/*.h"),
            ("aten/src", "ATen/native/vulkan/ops/*.h"),
            ("aten/src", "ATen/vulkan/*.h"),
        ]),
        feature = feature,
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "jit_core_headers",
        header_namespace = "",
        exported_headers = subdir_glob([("", x) for x in jit_core_headers]),
        feature = feature,
        labels = labels,
    )

    fb_xplat_cxx_library(
        name = "torch_headers",
        header_namespace = "",
        exported_headers = subdir_glob(
            [
                ("torch/csrc/api/include", "torch/**/*.h"),
                ("", "torch/csrc/**/*.h"),
                ("", "torch/csrc/generic/*.cpp"),
                ("", "torch/script.h"),
                ("", "torch/library.h"),
                ("", "torch/custom_class.h"),
                ("", "torch/custom_class_detail.h"),
                # Add again due to namespace difference from aten_header.
                ("", "aten/src/ATen/*.h"),
                ("", "aten/src/ATen/quantized/*.h"),
            ],
            exclude = [
                # Don't need on mobile.
                "torch/csrc/Exceptions.h",
                "torch/csrc/python_headers.h",
                "torch/csrc/utils/auto_gil.h",
                "torch/csrc/jit/serialization/mobile_bytecode_generated.h",
            ],
        ),
        feature = feature,
        labels = labels,
        visibility = ["PUBLIC"],
        deps = [
            ":generated-version-header",
        ],
    )

    fb_xplat_cxx_library(
        name = "aten_test_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            ("aten/src", "ATen/test/*.h"),
        ]),
    )

    fb_xplat_cxx_library(
        name = "torch_mobile_headers",
        header_namespace = "",
        exported_headers = subdir_glob(
            [
                ("", "torch/csrc/jit/mobile/*.h"),
            ],
        ),
        feature = feature,
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "generated_aten_config_header",
        header_namespace = "ATen",
        exported_headers = {
            "Config.h": ":generate_aten_config[Config.h]",
        },
        feature = feature,
        labels = labels,
    )

    fb_xplat_cxx_library(
        name = "generated-autograd-headers",
        header_namespace = "torch/csrc/autograd/generated",
        exported_headers = {
            "Functions.h": ":gen_aten_libtorch[autograd/generated/Functions.h]",
            "VariableType.h": ":gen_aten_libtorch[autograd/generated/VariableType.h]",
            "variable_factories.h": ":gen_aten_libtorch[autograd/generated/variable_factories.h]",
            # Don't build python bindings on mobile.
            #"python_functions.h",
        },
        feature = feature,
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "generated-version-header",
        header_namespace = "torch",
        exported_headers = {
            "version.h": ":generate-version-header[version.h]",
        },
        feature = feature,
        labels = labels,
    )

    # @lint-ignore BUCKLINT
    fb_native.genrule(
        name = "generate-version-header",
        srcs = [
            "torch/csrc/api/include/torch/version.h.in",
            "version.txt",
        ],
        cmd = "$(exe {}tools/setup_helpers:gen-version-header) ".format(ROOT_PATH) + " ".join([
            "--template-path",
            "torch/csrc/api/include/torch/version.h.in",
            "--version-path",
            "version.txt",
            "--output-path",
            "$OUT/version.h",
        ]),
        outs = {
            "version.h": ["version.h"],
        },
        default_outs = ["."],
    )

    fb_xplat_cxx_library(
        name = "generated_aten_headers_cpu",
        header_namespace = "ATen",
        exported_headers = get_aten_static_dispatch_backend_headers({
            "CPUFunctions.h": ":gen_aten[CPUFunctions.h]",
            "CPUFunctions_inl.h": ":gen_aten[CPUFunctions_inl.h]",
            "CompositeExplicitAutogradFunctions.h": ":gen_aten[CompositeExplicitAutogradFunctions.h]",
            "CompositeExplicitAutogradFunctions_inl.h": ":gen_aten[CompositeExplicitAutogradFunctions_inl.h]",
            "CompositeImplicitAutogradFunctions.h": ":gen_aten[CompositeImplicitAutogradFunctions.h]",
            "CompositeImplicitAutogradFunctions_inl.h": ":gen_aten[CompositeImplicitAutogradFunctions_inl.h]",
            "FunctionalInverses.h": ":gen_aten[FunctionalInverses.h]",
            "Functions.h": ":gen_aten[Functions.h]",
            "MethodOperators.h": ":gen_aten[MethodOperators.h]",
            "NativeFunctions.h": ":gen_aten[NativeFunctions.h]",
            "NativeMetaFunctions.h": ":gen_aten[NativeMetaFunctions.h]",
            "Operators.h": ":gen_aten[Operators.h]",
            "RedispatchFunctions.h": ":gen_aten[RedispatchFunctions.h]",
            "core/TensorBody.h": ":gen_aten[core/TensorBody.h]",
            "core/aten_interned_strings.h": ":gen_aten[core/aten_interned_strings.h]",
        }),
        feature = feature,
        labels = labels,
    )

    gen_aten_files(
        name = "gen_aten",
        extra_flags = get_aten_codegen_extra_params(USED_PT_BACKENDS),
        visibility = ["PUBLIC"],
    )

    # not used?
    gen_aten_files(
        name = "gen_aten_vulkan",
        extra_flags = get_aten_codegen_extra_params(USED_PT_BACKENDS),
        visibility = ["PUBLIC"],
    )

    # @lint-ignore BUCKLINT
    fb_native.filegroup(
        name = "aten_src_path",
        srcs = [
            "aten/src/ATen/native/native_functions.yaml",
            "aten/src/ATen/native/tags.yaml",
            # @lint-ignore BUCKRESTRICTEDSYNTAX
        ] + glob(["aten/src/ATen/templates/*"]),
        visibility = [
            "PUBLIC",
        ],
    )
