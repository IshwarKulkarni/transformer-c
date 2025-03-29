# Main variables
SRCDIR := src
INCDIR := headers
BUILDDIR := bin
OBJDIR := $(BUILDDIR)/obj
OBJDIR_EXE := $(BUILDDIR)/obj_exe

# Debug location
ifeq ($(dbg),1)
	OBJDIR = $(BUILDDIR)/dbg/obj
	OBJDIR_EXE = $(BUILDDIR)/dbg/obj_exe
endif

# Find all .cpp files in root directory
ROOT_CPP_FILES := $(wildcard *.cpp)
ROOT_TARGETS := $(patsubst %.cpp,$(BUILDDIR)/%,$(ROOT_CPP_FILES))
ROOT_OBJECTS := $(patsubst %.cpp,$(OBJDIR_EXE)/%.o,$(ROOT_CPP_FILES))

# Other variables
SOURCES := $(shell find $(SRCDIR) -type f -name "*.cpp")
OBJECTS := $(patsubst $(SRCDIR)/%,$(OBJDIR)/%,$(SOURCES:.cpp=.o))

SOURCESCU := $(shell find $(SRCDIR) -type f -name "*.cu")
OBJECTSCU := $(patsubst $(SRCDIR)/%,$(OBJDIR)/%,$(SOURCESCU:.cu=.cu.o))
TARGETDIR = `dirname $(BUILDDIR)`

# Compilers
HOST_COMPILER := g++-11
NVCC          := /usr/local/cuda-12.5/bin/nvcc -ccbin $(HOST_COMPILER)

# Flags
NVCCFLAGS     := -m64 
CCFLAGS       := --std=c++17 -fPIC -rdynamic -Wall -Wextra -Wsign-compare -I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11
LDFLAGS       :=

# Debug build flags
ifeq ($(dbg),1)
    NVCCFLAGS += -g -G
	CCFLAGS += -g -O0 -DDEBUG
else
	NVCCFLAGS += -lineinfo
# -DDISABLE_SIZE_CHECK
	CCFLAGS += -O3 
endif

# Main flags
ALL_CCFLAGS := $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_LDFLAGS := $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Includes and libraries
#INCLUDES  := $(addprefix -I ,$(shell find $(SRCDIR) -type d))
LIBRARIES := 

ifneq ($(INCDIR),)
	INCLUDES += -I $(INCDIR)
endif

ifeq ($(target),$(TARGET_TIMING) )
	ALL_CCFLAGS += -DDISABLE_SIZE_CHECK
endif

################################################################################

# Gencode arguments
SMS ?= 86

# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -arch=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
################################################################################

# Target rules
.PHONY: build clean run_% valgrind

.PRECIOUS: $(OBJDIR)/%.out $(OBJDIR_EXE)/%.o $(OBJDIR)/%.cu.o $(OBJDIR)/%.o

build: $(ROOT_TARGETS)

clean:
	rm -fr bin/* temp/* *.csv test.info

$(OBJDIR)/%.cu.o: $(SRCDIR)/%.cu
	@mkdir -p $(OBJDIR);
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -shared -c $< -o $@ 

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR);
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -shared -c  -dc $< -o $@

$(OBJDIR_EXE)/%.o: %.cpp
	@mkdir -p $(OBJDIR_EXE);
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

$(BUILDDIR)/%: $(OBJDIR_EXE)/%.o $(OBJECTSCU) $(OBJECTS)
	@mkdir -p $(TARGETDIR);
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $+ $(LIBRARIES) -o $@
	@echo "\033[1;32mBuild complete for $@ \033[0m"

# Run targets for each executable
run_%: $(BUILDDIR)/%
	./$< $(var)

valgrind: build
	valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --gen-suppressions=all \
         --suppressions=cuda_supp.sup ./$(BUILDDIR)/main
