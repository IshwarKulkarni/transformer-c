# Main variables
SRCDIR := src
INCDIR := headers
BUILDDIR := bin
OBJDIR := $(BUILDDIR)/obj
TARGET_MAIN := bin/main
TARGET_TEST := bin/test

ALL_TARGETS := $(TARGET_MAIN) $(TARGET_TEST)

# Other variables
SOURCES := $(shell find $(SRCDIR) -type f -name "*.cpp")
# remove "main.cpp" and "tests.cpp" from SOURCES

OBJECTS := $(patsubst $(SRCDIR)/%,$(OBJDIR)/%,$(SOURCES:.cpp=.o))

# remove "main.o" from test objects and "test.o" from main objects
OBJECTS_MAIN := $(filter-out $(OBJDIR)/tests.o, $(OBJECTS))  
OBJECTS_TEST := $(filter-out $(OBJDIR)/main.o, $(OBJECTS))

SOURCESCU := $(shell find $(SRCDIR) -type f -name "*.cu")
OBJECTSCU := $(patsubst $(SRCDIR)/%,$(OBJDIR)/%,$(SOURCESCU:.cu=.cu.o))
TARGETDIR = `dirname $(BUILDDIR)`

# Compilers
HOST_COMPILER := g++-9
NVCC          := nvcc -ccbin $(HOST_COMPILER)  #/usr/local/cuda/bin/nvcc

# Flags
NVCCFLAGS     := -m64
CCFLAGS       := --std=c++14 -fPIC -rdynamic -Wall -Wextra -Wno-sign-compare
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

build: $(ALL_TARGETS)

clean:
	rm -fr $(OBJDIR) $(OBJECTSCU) $(ALL_TARGETS) *.csv temp/*

$(OBJDIR)/%.cu.o: $(SRCDIR)/%.cu
	@mkdir -p $(OBJDIR);
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -shared -c $< -o $@ 

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR);
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -shared -c $< -o $@

$(TARGET_MAIN): $(OBJECTS_MAIN) $(OBJECTSCU)
	@mkdir -p $(TARGETDIR);
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $+ $(LIBRARIES) -o $@ 
	@echo "\033[1;32mBuild complete for $(TARGET_MAIN) \033[0m "

$(TARGET_TEST): $(OBJECTS_TEST) $(OBJECTSCU)
	@mkdir -p $(TARGETDIR);
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $+ $(LIBRARIES) -o $@
	@echo "\033[1;32mBuild complete for $(TARGET_TEST) \033[0m "

valgrind: build
	valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --gen-suppressions=all \
         --suppressions=cuda_supp.sup ./$(TARGET_MAIN)
