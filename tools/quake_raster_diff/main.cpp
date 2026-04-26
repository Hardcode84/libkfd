#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr std::uintmax_t MAX_COMPARE_BYTES = 1024ULL * 1024ULL * 1024ULL;

bool read_file(const std::string &path, std::vector<std::uint8_t> &out) {
  std::error_code ec;
  const std::uintmax_t size = std::filesystem::file_size(path, ec);
  if (ec) {
    std::cerr << "failed to stat " << path << ": " << ec.message() << '\n';
    return false;
  }
  if (size > MAX_COMPARE_BYTES) {
    std::cerr << "refusing to load " << path << " (" << size
              << " bytes; max " << MAX_COMPARE_BYTES << ")\n";
    return false;
  }

  std::ifstream file(path, std::ios::binary);
  if (!file.good()) {
    std::cerr << "failed to open " << path << '\n';
    return false;
  }
  out.reserve(static_cast<std::size_t>(size));
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
  std::size_t first_difference = std::numeric_limits<std::size_t>::max();
  int max_delta = 0;
  std::uint64_t absolute_error = 0;

  for (std::size_t i = 0; i < shared; ++i) {
    const int delta =
        static_cast<int>(expected[i]) - static_cast<int>(actual[i]);
    if (delta != 0) {
      if (first_difference == std::numeric_limits<std::size_t>::max()) {
        first_difference = i;
      }
      ++different;
      const int magnitude = delta < 0 ? -delta : delta;
      max_delta = std::max(max_delta, magnitude);
      absolute_error += static_cast<std::uint64_t>(magnitude);
    }
  }
  if (first_difference == std::numeric_limits<std::size_t>::max() &&
      expected.size() != actual.size()) {
    first_difference = shared;
  }

  std::cout << "bytes_expected=" << expected.size()
            << " bytes_actual=" << actual.size() << " different=" << different
            << " absolute_error=" << absolute_error
            << " max_delta=" << max_delta
            << " first_difference="
            << (first_difference == std::numeric_limits<std::size_t>::max()
                    ? -1
                    : static_cast<long long>(first_difference))
            << '\n';
  return different == 0U ? 0 : 1;
}
