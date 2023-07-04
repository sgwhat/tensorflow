"""loads the jpeg library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive_new", "tf_mirror_urls", third_party_http_archive)

def repo():
    third_party_http_archive(
        name = "jpeg",
        strip_prefix = "libjpeg-turbo-2.1.4",
        sha256 = "a78b05c0d8427a90eb5b4eb08af25309770c8379592bb0b8a863373128e6143f",
        urls = [
                "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/2.1.4.tar.gz",
                "https://storage.googleapis.com/mirror.tensorflow.org/github.com/libjpeg-turbo/libjpeg-turbo/archive/2.1.4.tar.gz",
                ],
        build_file = "//third_party/jpeg:BUILD.bazel",
        system_build_file = "//third_party/jpeg:BUILD.system",
    )