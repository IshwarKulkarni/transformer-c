# Main variables
SRCDIR := src
INCDIR := include
BUILDDIR := bin
TARGET := bin/main

# Other variables
SOURCES := $(shell find $(SRCDIR) -type f -name "*.cpp")
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.cpp=.o))
SOURCESCU := $(shell find $(SRCDIR) -type f -name "*.cu")
OBJECTSCU := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCESCU:.cu=.cu.o))
TARGETDIR = `dirname $(TARGET)`

# Compilers
HOST_COMPILER := g++
NVCC          := nvcc -ccbin $(HOST_COMPILER)  #/usr/local/cuda/bin/nvcc

# Flags
NVCCFLAGS     := -m64  #-dc used cg::grid_group, cg::this_group
CCFLAGS       := --std=c++0x -fPIC
LDFLAGS       :=

# Debug build flags
ifeq ($(dbg),1)
    NVCCFLAGS += -g -G
    BUILD_TYPE := debug
	CCFLAGS += -g -O0 -DDEBUG
else
    BUILD_TYPE := release
	CCFLAGS += -O3
endif

# Main flags
ALL_CCFLAGS := $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_LDFLAGS := $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Includes and libraries
INCLUDES      := $(addprefix -I ,$(shell find $(SRCDIR) -type d))
LIBRARIES     := 

ifneq ($(INCDIR),)
	INCLUDES += -I $(INCDIR)
endif

################################################################################

# Gencode arguments
# SMS ?= 30 35 37 50 52 60 61 70
SMS ?= 70

# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif

################################################################################

# Target rules
all: build

build: $(TARGET)

clean:
	rm -fr $(OBJECTS) $(OBJECTSCU) $(TARGET) *.csv

$(BUILDDIR)/%.cu.o: $(SRCDIR)/%.cu
	@mkdir -p $(BUILDDIR);
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -shared -c $< -o $@ 

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR);
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -shared -c $< -o $@

$(TARGET): $(OBJECTS) $(OBJECTSCU)
	@mkdir -p $(TARGETDIR);
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $+ $(LIBRARIES) -o $@ 