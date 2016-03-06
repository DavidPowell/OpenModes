# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
#  OpenModes - An eigenmode solver for open electromagnetic resonantors
#  Copyright (C) 2013 David Powell
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#-----------------------------------------------------------------------------

"""This file contains the version number of OpenModes. It will be called by
setup.py before installing, generating source package etc.

If the source is controlled by git, then git will be used to get the latest
tag and any subsequent updates.

Otherwise, a hard-coded value will be used - this must be UPDATED MANUALLY
before tagging each release

Based on public domain code from Douglas Creager
"""

from subprocess import Popen, PIPE

# THIS MUST BE UPDATED MANUALLY FOR NON-GIT
RELEASE_VERSION = "1.0.0"


def version_git():
    try:
        p = Popen(['git', 'describe', '--abbrev=%d' % 4], stdout=PIPE,
                  stderr=PIPE, universal_newlines=True)
        p.stderr.close()
        line = p.stdout.readlines()[0]
        return line.strip()
    except:
        return None

__version__ = version_git() or RELEASE_VERSION
