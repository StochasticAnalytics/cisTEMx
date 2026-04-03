add_executable(generic_runner generic_runner.cpp)
add_dependencies(generic_runner cisTEM_core)

target_link_libraries(generic_runner cisTEM_core)

install(TARGETS generic_runner
        RUNTIME DESTINATION bin)
