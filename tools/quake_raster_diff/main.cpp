#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

bool read_file(const std::string &path, std::vector<std::uint8_t> &out) {
  std::ifstream file(path, std::ios::binary);
  if (!file.good()) {
    std::cerr << "failed to open " << path << '\n';
    return false;
  }
  out.assign(std::istreambuf_iterator<char>(file),
             std::istreambuf_iterator<char>());
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: quake_raster_diff <expected.raw> <actual.raw>\n";
    return 2;
  }

  std::vector<std::uint8_t> expected;
  std::vector<std::uint8_t> actual;
  if (!read_file(argv[1], expected) || !read_file(argv[2], actual)) {
    return 2;
  }
  const std::size_t shared = std::min(expected.size(), actual.size());
  std::size_t different =
      expected.size() == actual.size()
          ? 0U
          : std::max(expected.size(), actual.size()) - shared;
  std::uint64_t absolute_error = 0;

  for (std::size_t i = 0; i < shared; ++i) {
    const int delta =
        static_cast<int>(expected[i]) - static_cast<int>(actual[i]);
    if (delta != 0) {
      ++different;
      absolute_error += static_cast<std::uint64_t>(delta < 0 ? -delta : delta);
    }
  }

  std::cout << "bytes_expected=" << expected.size()
            << " bytes_actual=" << actual.size() << " different=" << different
            << " absolute_error=" << absolute_error << '\n';
  return different == 0U ? 0 : 1;
}
