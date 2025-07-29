from pathlib import Path

import logging

# Import JPype Module for Java imports
import jpype.imports
from jpype.types import *

import os

java_home = os.environ.get("JAVA_HOME", None)
print(f'java_home: {java_home}')
print(f'jvm_version: {jpype.getJVMVersion()}')

# jvm_path = None if java_home is None else str(Path(java_home) / "lib/server/libjvm.dylib")
# print(f'jvm_path: {jvm_path}')
print(f'default jvm_path: {jpype.getDefaultJVMPath()}')


DEFAULT_JAVA_PATH = Path(__file__).parent.parent.parent.parent.parent / "lib"
# DEFAULT_JAVA_PATH = Path(__file__).parent.parent.parent.parent.parent / "build/lib"
# DEFAULT_JAVA_PATH = Path(java_home) / "lib"
# DEFAULT_JAVA_PATH = Path(java_home) / "bin"

java_path = str(Path(os.environ.get("JAVA_LIB", DEFAULT_JAVA_PATH)) / "*")
# java_path = str(Path(os.environ.get("JAVA_LIB", DEFAULT_JAVA_PATH)) / "random-cut-forest-1.0.jar")
# java_path = str(Path(os.environ.get("JAVA_LIB", DEFAULT_JAVA_PATH)) / "java")
print(f'java_path: {java_path}')
print (f'default class path: {jpype.getClassPath()}')

jpype.addClassPath(java_path)

# Launch the JVM
# jpype.startJVM("-XX:-UseContainerSupport")
jpype.startJVM(convertStrings=False)

#logging.info("Started JVM with version %s", jpype.getJVMVersion())
logging.info("availableProcess {}".format(jpype.java.lang.Runtime.getRuntime().availableProcessors()))
