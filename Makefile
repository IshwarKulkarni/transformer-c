# Main variables
SRCDIR := src
INCDIR := headers
BUILDDIR := bin
OBJDIR := $(BUILDDIR)/obj
TARGET_MAIN := $(BUILDDIR)/main
TARGET_TEST := $(BUILDDIR)/test
TARGET_TIMING := $(BUILDDIR)/timing

# Debug location
ifeq ($(dbg),1)
	OBJDIR = $(BUILDDIR)/dbg/obj
	TARGET_MAIN = bin/dbg/main
	TARGET_TEST = bin/dbg/test
	TARGET_TIMING = bin/dbg/timing
endif

ALL_TARGETS := $(TARGET_MAIN) $(TARGET_TEST) $(TARGET_TIMING)

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
CCFLAGS       := --std=c++17 -fPIC -rdynamic -Wall -Wextra -Wsign-compare
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
# SMS ?= 30 35 37 50 52 60 61 70
SMS ?= 86

# Generate SASS code for each SM architecture listed in $(SMS)
#$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -arch=compute_$(HIGHEST_SM) -code=compute_$(HIGHEST_SM) -allow-unsupported-compiler
endif

################################################################################

# Target rules
build: $(ALL_TARGETS)

clean:
	rm -fr bin/* temp/* *.csv test.info

$(OBJDIR)/%.cu.o: $(SRCDIR)/%.cu
	@mkdir -p $(OBJDIR);
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -shared -c $< -o $@ 

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR);
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -shared -c  -dc $< -o $@

$(TARGET_MAIN): $(OBJECTSCU) $(OBJDIR)/main.o
	@mkdir -p $(TARGETDIR);
	@touch src/main.cpp
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $+ $(LIBRARIES) -o $@ 
	@echo "\033[1;32mBuild complete for $(TARGET_MAIN) \033[0m "

$(TARGET_TEST): $(OBJECTSCU) $(OBJDIR)/test.o
	@mkdir -p $(TARGETDIR);
	@touch src/test.cpp
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $+ $(LIBRARIES) -o $@
	@echo "\033[1;32mBuild complete for $(TARGET_TEST) \033[0m "

$(TARGET_TIMING): $(OBJECTSCU) $(OBJDIR)/timing.o
	@mkdir -p $(TARGETDIR);
	@touch src/timing.cpp
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $+ $(LIBRARIES) -o $@
	@echo "\033[1;32mBuild complete for $(TARGET_TIMING) \033[0m"

# alias main, timing and test
run_main: $(TARGET_MAIN)
	./$(TARGET_MAIN) $(var)

run_test: $(TARGET_TEST)
	./$(TARGET_TEST) $(var)

run_timing: $(TARGET_TIMING)
	./$(TARGET_TIMING) $(var)

valgrind: build
	valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --gen-suppressions=all \
         --suppressions=cuda_supp.sup ./$(TARGET_MAIN)
