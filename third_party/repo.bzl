# Copyright 2017 The TensorFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for defining TensorFlow Bazel dependencies."""

_SINGLE_URL_WHITELIST = depset([
    "arm_compiler",
])

def tf_mirror_urls(url):
    """A helper for generating TF-mirror versions of URLs.

    Given a URL, it returns a list of the TF-mirror cache version of that URL
    and the original URL, suitable for use in `urls` field of `tf_http_archive`.
    """
    if not url.startswith("https://"):
        return [url]
    return [
        "https://storage.googleapis.com/mirror.tensorflow.org/%s" % url[8:],
        url,
    ]

def _is_windows(ctx):
    return ctx.os.name.lower().find("windows") != -1

def _wrap_bash_cmd(ctx, cmd):
    if _is_windows(ctx):
        bazel_sh = _get_env_var(ctx, "BAZEL_SH")
        if not bazel_sh:
            fail("BAZEL_SH environment variable is not set")
        cmd = [bazel_sh, "-l", "-c", " ".join(["\"%s\"" % s for s in cmd])]
    return cmd

def _get_env_var(ctx, name):
    if name in ctx.os.environ:
        return ctx.os.environ[name]
    else:
        return None

# Checks if we should use the system lib instead of the bundled one
def _use_system_lib(ctx, name):
    syslibenv = _get_env_var(ctx, "TF_SYSTEM_LIBS")
    if syslibenv:
        for n in syslibenv.strip().split(","):
            if n.strip() == name:
                return True
    return False

def _get_link_dict(ctx, link_files, build_file):
    link_dict = {ctx.path(v): ctx.path(Label(k)) for k, v in link_files.items()}
    if build_file:
        # Use BUILD.bazel because it takes precedence over BUILD.
        link_dict[ctx.path("BUILD.bazel")] = ctx.path(Label(build_file))
    return link_dict

def _tf_http_archive_impl(ctx):
    # Construct all paths early on to prevent rule restart. We want the
    # attributes to be strings instead of labels because they refer to files
    # in the TensorFlow repository, not files in repos depending on TensorFlow.
    # See also https://github.com/bazelbuild/bazel/issues/10515.
    link_dict = _get_link_dict(ctx, ctx.attr.link_files, ctx.attr.build_file)

    # For some reason, we need to "resolve" labels once before the
    # download_and_extract otherwise it'll invalidate and re-download the
    # archive each time.
    # https://github.com/bazelbuild/bazel/issues/10515
    patch_files = ctx.attr.patch_file
    for patch_file in patch_files:
        if patch_file:
            ctx.path(Label(patch_file))

    if _use_system_lib(ctx, ctx.attr.name):
        link_dict.update(_get_link_dict(
            ctx = ctx,
            link_files = ctx.attr.system_link_files,
            build_file = ctx.attr.system_build_file,
        ))
    else:
        ctx.download_and_extract(
            url = ctx.attr.urls,
            sha256 = ctx.attr.sha256,
            type = ctx.attr.type,
            stripPrefix = ctx.attr.strip_prefix,
        )
        if patch_files:
            for patch_file in patch_files:
                patch_file = ctx.path(Label(patch_file)) if patch_file else None
                if patch_file:
                    ctx.patch(patch_file, strip = 1)

    for dst, src in link_dict.items():
        ctx.delete(dst)
        ctx.symlink(src, dst)

# Executes specified command with arguments and calls 'fail' if it exited with
# non-zero code
def _execute_and_check_ret_code(repo_ctx, cmd_and_args):
    result = repo_ctx.execute(cmd_and_args, timeout = 60)
    if result.return_code != 0:
        fail(("Non-zero return code({1}) when executing '{0}':\n" + "Stdout: {2}\n" +
              "Stderr: {3}").format(
            " ".join(cmd_and_args),
            result.return_code,
            result.stdout,
            result.stderr,
        ))

def _repos_are_siblings():
    return Label("@foo//bar").workspace_root.startswith("../")

# Apply a patch_file to the repository root directory
# Runs 'git apply' on Unix, 'patch -p1' on Windows.
def _apply_patch(ctx, patch_file):
    if _is_windows(ctx):
        patch_command = ["patch", "-p1", "-d", ctx.path("."), "-i", ctx.path(patch_file)]
    else:
        patch_command = ["git", "apply", "-v", ctx.path(patch_file)]
    cmd = _wrap_bash_cmd(ctx, patch_command)
    _execute_and_check_ret_code(ctx, cmd)

def _apply_delete(ctx, paths):
    for path in paths:
        if path.startswith("/"):
            fail("refusing to rm -rf path starting with '/': " + path)
        if ".." in path:
            fail("refusing to rm -rf path containing '..': " + path)
    cmd = _wrap_bash_cmd(ctx, ["rm", "-rf"] + [ctx.path(path) for path in paths])
    _execute_and_check_ret_code(ctx, cmd)

_tf_http_archive_new = repository_rule(
    implementation = _tf_http_archive_impl,
    attrs = {
        "sha256": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True),
        "strip_prefix": attr.string(),
        "type": attr.string(),
        "patch_file": attr.string_list(),
        "build_file": attr.string(),
        "system_build_file": attr.string(),
        "link_files": attr.string_dict(),
        "system_link_files": attr.string_dict(),
    },
    environ = ["TF_SYSTEM_LIBS"],
)

def tf_http_archive_new(name, sha256, urls, **kwargs):
    """Downloads and creates Bazel repos for dependencies.

    This is a swappable replacement for both http_archive() and
    new_http_archive() that offers some additional features. It also helps
    ensure best practices are followed.

    File arguments are relative to the TensorFlow repository by default. Dependent
    repositories that use this rule should refer to files either with absolute
    labels (e.g. '@foo//:bar') or from a label created in their repository (e.g.
    'str(Label("//:bar"))').
    """
    if len(urls) < 2:
        fail("tf_http_archive(urls) must have redundant URLs.")

    if not any([mirror in urls[0] for mirror in (
        "mirror.tensorflow.org",
        "mirror.bazel.build",
        "storage.googleapis.com",
    )]):
        fail("The first entry of tf_http_archive(urls) must be a mirror " +
             "URL, preferrably mirror.tensorflow.org. Even if you don't have " +
             "permission to mirror the file, please put the correctly " +
             "formatted mirror URL there anyway, because someone will come " +
             "along shortly thereafter and mirror the file.")

    if native.existing_rule(name):
        print("\n\033[1;33mWarning:\033[0m skipping import of repository '" +
              name + "' because it already exists.\n")
        return

    _tf_http_archive_new(
        name = name,
        sha256 = sha256,
        urls = urls,
        **kwargs
    )


def _tf_http_archive(ctx):
    if ("mirror.tensorflow.org" not in ctx.attr.urls[0] and
        (len(ctx.attr.urls) < 2 and
         ctx.attr.name not in _SINGLE_URL_WHITELIST.to_list())):
        fail("tf_http_archive(urls) must have redundant URLs. The " +
             "mirror.tensorflow.org URL must be present and it must come first. " +
             "Even if you don't have permission to mirror the file, please " +
             "put the correctly formatted mirror URL there anyway, because " +
             "someone will come along shortly thereafter and mirror the file.")

    use_syslib = _use_system_lib(ctx, ctx.attr.name)
    if not use_syslib:
        ctx.download_and_extract(
            ctx.attr.urls,
            "",
            ctx.attr.sha256,
            ctx.attr.type,
            ctx.attr.strip_prefix,
        )
        if ctx.attr.delete:
            _apply_delete(ctx, ctx.attr.delete)
        if ctx.attr.patch_file != None:
            _apply_patch(ctx, ctx.attr.patch_file)

    if use_syslib and ctx.attr.system_build_file != None:
        # Use BUILD.bazel to avoid conflict with third party projects with
        # BUILD or build (directory) underneath.
        ctx.template("BUILD.bazel", ctx.attr.system_build_file, {
            "%prefix%": ".." if _repos_are_siblings() else "external",
        }, False)

    elif ctx.attr.build_file != None:
        # Use BUILD.bazel to avoid conflict with third party projects with
        # BUILD or build (directory) underneath.
        ctx.template("BUILD.bazel", ctx.attr.build_file, {
            "%prefix%": ".." if _repos_are_siblings() else "external",
        }, False)

    if use_syslib:
        for internal_src, external_dest in ctx.attr.system_link_files.items():
            ctx.symlink(Label(internal_src), ctx.path(external_dest))

tf_http_archive = repository_rule(
    implementation = _tf_http_archive,
    attrs = {
        "sha256": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True, allow_empty = False),
        "strip_prefix": attr.string(),
        "type": attr.string(),
        "delete": attr.string_list(),
        "patch_file": attr.label(),
        "build_file": attr.label(),
        "system_build_file": attr.label(),
        "system_link_files": attr.string_dict(),
    },
    environ = [
        "TF_SYSTEM_LIBS",
    ],
)
"""Downloads and creates Bazel repos for dependencies.

This is a swappable replacement for both http_archive() and
new_http_archive() that offers some additional features. It also helps
ensure best practices are followed.
"""

def _third_party_http_archive(ctx):
    if ("mirror.tensorflow.org" not in ctx.attr.urls[0] and
        (len(ctx.attr.urls) < 2 and
         ctx.attr.name not in _SINGLE_URL_WHITELIST.to_list())):
        fail("tf_http_archive(urls) must have redundant URLs. The " +
             "mirror.tensorflow.org URL must be present and it must come first. " +
             "Even if you don't have permission to mirror the file, please " +
             "put the correctly formatted mirror URL there anyway, because " +
             "someone will come along shortly thereafter and mirror the file.")

    use_syslib = _use_system_lib(ctx, ctx.attr.name)

    # Use "BUILD.bazel" to avoid conflict with third party projects that contain a
    # file or directory called "BUILD"
    buildfile_path = ctx.path("BUILD.bazel")

    if use_syslib:
        if ctx.attr.system_build_file == None:
            fail("Bazel was configured with TF_SYSTEM_LIBS to use a system " +
                 "library for %s, but no system build file for %s was configured. " +
                 "Please add a system_build_file attribute to the repository rule" +
                 "for %s." % (ctx.attr.name, ctx.attr.name, ctx.attr.name))
        ctx.symlink(Label(ctx.attr.system_build_file), buildfile_path)

    else:
        ctx.download_and_extract(
            ctx.attr.urls,
            "",
            ctx.attr.sha256,
            ctx.attr.type,
            ctx.attr.strip_prefix,
        )
        if ctx.attr.delete:
            _apply_delete(ctx, ctx.attr.delete)
        if ctx.attr.patch_file != None:
            _apply_patch(ctx, ctx.attr.patch_file)
        ctx.symlink(Label(ctx.attr.build_file), buildfile_path)

    link_dict = {}
    if use_syslib:
        link_dict.update(ctx.attr.system_link_files)

    for internal_src, external_dest in ctx.attr.link_files.items():
        # if syslib and link exists in both, use the system one
        if external_dest not in link_dict.values():
            link_dict[internal_src] = external_dest

    for internal_src, external_dest in link_dict.items():
        ctx.symlink(Label(internal_src), ctx.path(external_dest))

# Downloads and creates Bazel repos for dependencies.
#
# This is an upgrade for tf_http_archive that works with go/tfbr-thirdparty.
#
# For link_files, specify each dict entry as:
# "//path/to/source:file": "localfile"
third_party_http_archive = repository_rule(
    implementation = _third_party_http_archive,
    attrs = {
        "sha256": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True, allow_empty = False),
        "strip_prefix": attr.string(),
        "type": attr.string(),
        "delete": attr.string_list(),
        "build_file": attr.string(mandatory = True),
        "system_build_file": attr.string(mandatory = False),
        "patch_file": attr.label(),
        "link_files": attr.string_dict(),
        "system_link_files": attr.string_dict(),
    },
    environ = [
        "TF_SYSTEM_LIBS",
    ],
)
