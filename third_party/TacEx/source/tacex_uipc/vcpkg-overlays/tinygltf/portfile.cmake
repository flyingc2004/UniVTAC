# Pin an immutable upstream revision.  GitHub's mutable v2.9.4 tag archive no
# longer matches the historical checksum in the selected vcpkg baseline.
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO syoyo/tinygltf
    REF "5f330f39528aa32285a5d51cf94f7fa326e8baff"
    SHA512 e96fc8db91dfbf86eeed4c93f70d5d60425391a2d2b21e42c7c9c1746ce7b7ed39f2623314ccce0ce3c13b3beece9b17e766acd0efabb8756729a835340c6af3
    HEAD_REF master
)

vcpkg_replace_string("${SOURCE_PATH}/tiny_gltf.h" "#include \"json.hpp\"" "#include <nlohmann/json.hpp>")
file(INSTALL "${SOURCE_PATH}/tiny_gltf.h" DESTINATION "${CURRENT_PACKAGES_DIR}/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
