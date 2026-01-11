# SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

CUDA_VER?=12.6
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

WSL:=$(shell uname -a | grep -io "microsoft" | tr '[:upper:]' '[:lower:]')

CXX:= g++
NVCC:= /usr/local/cuda-$(CUDA_VER)/bin/nvcc
SRCS:= gstnvinfer.cpp  gstnvinfer_allocator.cpp gstnvinfer_property_parser.cpp \
       gstnvinfer_meta_utils.cpp gstnvinfer_impl.cpp gstnvinfer_yaml_parser.cpp
INCS:= $(wildcard *.h) $(wildcard *.hpp)
LIB:=libnvdsgst_infer.so

NVDS_VERSION:=7.1
CFLAGS+= -fPIC -std=c++11 -fpermissive -DDS_VERSION=\"7.1.0\" \
     -I /usr/local/cuda-$(CUDA_VER)/include \
     -I /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/includes \
     -I /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/gst-plugins/gst-nvdspreprocess/include \
     -I /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/libs/nvdsinfer -DNDEBUG \
     -DWITH_OPENCV \
     -I /usr/include/opencv4

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lnppc -lnppig -lnppidei

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_customhelper -lnvdsgst_meta -lnvds_meta \
       -lnvds_infer -lnvbufsurface -lnvbufsurftransform -ldl -lpthread -lyaml-cpp \
       -lcuda -Wl,-rpath,$(LIB_INSTALL_DIR) \
       -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

ifeq ("microsoft", "$(WSL)")
  LIBS+= -L/usr/lib/wsl/lib/
endif

OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ $(NVCCFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
