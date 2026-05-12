file(GLOB SRC CONFIGURE_DEPENDS *.cu)

foreach(ITEM IN LISTS SRC)
    get_filename_component(EXE_TARGET "${ITEM}" NAME_WE)
    add_executable(${EXE_TARGET} ${ITEM})
endforeach()
