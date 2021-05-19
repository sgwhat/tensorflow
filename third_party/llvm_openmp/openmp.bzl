"""This file contains BUILD extensions for building llvm_openmp.
TODO(Intel-tf): Delete this and reuse a similar function in third_party/llvm
after the TF 2.4 branch cut has passed.
"""

load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_binary",
)

#WINDOWS_MSVC_LLVM_OPENMP_LIBPATH = "bazel-out/x64_windows-opt/bin/external/llvm_openmp/libiomp5md.dll.if.lib"
WINDOWS_MSVC_LLVM_OPENMP_LIBPATH = "$(BINDIR)/external/llvm_openmp/libiomp5md.dll.if.lib"
WINDOWS_MSVC_LLVM_OPENMP_LINKOPTS = "/NODEFAULTLIB:libomp /DEFAULTLIB:" + WINDOWS_MSVC_LLVM_OPENMP_LIBPATH

def windows_llvm_openmp_deps():
    return ["@llvm_openmp//:libiomp5md.dll"]


def windows_llvm_openmp_linkopts():
    return WINDOWS_MSVC_LLVM_OPENMP_LINKOPTS

def dict_add(*dictionaries):
    """Returns a new `dict` that has all the entries of the given dictionaries.

    If the same key is present in more than one of the input dictionaries, the
    last of them in the argument list overrides any earlier ones.

    Args:
      *dictionaries: Zero or more dictionaries to be added.

    Returns:
      A new `dict` that has all the entries of the given dictionaries.
    """
    result = {}
    for d in dictionaries:
        result.update(d)
    return result

def select_os_specific(L, M, W):
    return select({
        "@org_tensorflow//tensorflow:linux_x86_64": L,
        "@org_tensorflow//tensorflow:macos": M,
        "@org_tensorflow//tensorflow:windows": W,
        "//conditions:default": L,
    })

def libname_os_specific():
    return "" + select_os_specific(L = "libiomp5.so", M = "libiomp5.dylib", W = "libiomp5md.dll")

def libiomp5_cc_binary(libname, cppsources, srcdeps, common_includes):
    cc_binary(
        name = libname,
        srcs = cppsources + srcdeps +
            select_os_specific(
            L = [
                #linux & macos specific files
                "runtime/src/z_Linux_util.cpp",
                "runtime/src/kmp_gsupport.cpp",
                "runtime/src/z_Linux_asm.S",
            ],
            M = [
                #linux & macos specific files
                "runtime/src/z_Linux_util.cpp",
                "runtime/src/kmp_gsupport.cpp",
                "runtime/src/z_Linux_asm.S",
            ],
            W = [
                #window specific files
                "runtime/src/z_Windows_NT_util.cpp",
                "runtime/src/z_Windows_NT-586_util.cpp",
                ":openmp_asm",
            ]),
        copts = select_os_specific(
            L = ["-Domp_EXPORTS -D_GNU_SOURCE -D_REENTRANT"],
            M = ["-Domp_EXPORTS -D_GNU_SOURCE -D_REENTRANT"],
            W = ["/Domp_EXPORTS /D_M_AMD64 /DOMPT_SUPPORT=0 /D_WINDOWS /D_WINNT /D_USRDLL"],
        ),
        includes = common_includes,
        linkopts = select_os_specific(
            L = ["-lpthread -ldl -Wl,--version-script=$(location :ldscript)"],
            M = ["-lpthread -ldl -Wl,--version-script=$(location :ldscript)"],
            W = ["/MACHINE:X64"],
        ),
        linkshared = True,
        additional_linker_inputs = [":generate_def"],
        win_def_file = ":generate_def",
        visibility = ["//visibility:public"],
    )




