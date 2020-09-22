#bazel build --define framework_shared_object=false -s --verbose_failures -c opt  --config=cuda  //tensorflow/tools/pip_package:build_pip_package
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
