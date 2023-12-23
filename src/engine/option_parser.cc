/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021-2023  Connor McMonigle

  Seer is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Seer is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <engine/option_parser.h>

namespace engine {

std::ostream& operator<<(std::ostream& ostr, const string_option& opt) noexcept {
  ostr << "option name " << opt.name_ << " type string";
  if (opt.default_.has_value()) { ostr << " default " << opt.default_.value(); }
  return ostr;
}

std::ostream& operator<<(std::ostream& ostr, const spin_option& opt) noexcept {
  ostr << "option name " << opt.name_ << " type spin";
  if (opt.default_.has_value()) { ostr << " default " << opt.default_.value(); }
  if (opt.range_.has_value()) { ostr << " min " << opt.range_.value().min << " max " << opt.range_.value().max; }
  return ostr;
}

std::ostream& operator<<(std::ostream& ostr, const check_option& opt) noexcept {
  ostr << "option name " << opt.name_ << " type check";
  if (opt.default_.has_value()) { ostr << std::boolalpha << " default " << opt.default_.value(); }
  return ostr;
}

std::ostream& operator<<(std::ostream& ostr, const float_option& opt) noexcept {
  ostr << "option name " << opt.name_ << " type string";
  if (opt.default_.has_value()) { ostr << " default " << opt.default_.value(); }
  return ostr;
}

}  // namespace engine
