# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ["Log"]

import sys
import time
import traceback
import textwrap

import numpy as np


class Log():
  """
  Dual file/stdout logging class with conditional printing.
  """
  def __init__(self, logname, verb=1, append=False, width=70):
    """
    Parameters
    ----------
    logname: String
       Name of FILE pointer where to store log entries. Set to None to
       print only to stdout.
    verb: Integer
       Conditional threshold to print the message.  Print only if
       verblevel is positive.
    append: Bool
       If true, do not overrite logname file if it already exists.
    width: Integer
       Maximum length of each line of text (longer texts will be break
       down into multiple lines).
    """
    #self.logname = os.path.realpath(logname)
    self.logname = logname
    if self.logname is not None:
      if append:
        self.file = open(self.logname, "aw")
      else:
        self.file = open(self.logname, "w")
    else:
      self.file = None
    self.verb = verb
    self.indent = 0
    self.width  = width
    self.warnings = []
    self.sep = 70*":"  # Warning separator


  def write(self, text):
    """
    Write and flush text to stdout and FILE pointer if it exists.

    Parameters
    ----------
    text: String
       Text to write.
    """
    # Print to screen and file:
    print(text)
    sys.stdout.flush()
    # Print to file, if requested:
    if self.file is not None:
      self.file.write(text + "\n")
      self.file.flush()


  def msg(self, message, verb=1, indent=None, noprint=False,
          si=None, width=None):
    """
    Conditional message printing to screen and to file.

    Parameters
    ----------
    message: String
       String to be printed.
    verb: Integer
       Required verbosity level: print only if self.verb >= verb.
    indent: Integer
       Number of blank spaces to indent the printed message.
    noprint: Boolean
       If True, do not print and return the string instead.
    si: Integer
       Sub-sequent lines indentation.
    width: Integer
       If not None, override text width (only for this specific call).

    Returns
    -------
    text: String
       If noprint is True, return the formatted output string.
    """
    if self.verb < verb:
      return

    if indent is None:
      indent = self.indent
    if si is None:
      si = self.indent
    if width is None:
      width = self.width

    # Indentation texts:
    indspace = " "*indent
    sind     = " "*si

    # Break down the input text into the different sentences (line-breaks):
    sentences = message.splitlines()

    # Output text to be printed:
    msg = []
    for sentence in sentences:
      msg.append(textwrap.fill(sentence,
                               break_long_words=False,
                               break_on_hyphens=False,
                               initial_indent=indspace,
                               subsequent_indent=sind,
                               width=width))
    text = "\n".join(msg)

    # Do not print, just return the string:
    if noprint:
      return text
    # Print to screen and file:
    self.write(text)


  def warning(self, message):
    """
    Print a warning message surrounded by colon bands.

    Parameters
    ----------
    message: String
       String to be printed.
    """
    if self.verb < 0:
      return

    # Format the sub-text message:
    subtext = self.msg(message, verb=1, indent=4, noprint=True)
    # Add the warning surroundings:
    text = ("\n{:s}"
            "\n  Warning:"
            "\n{:s}"
            "\n{:s}\n".format(self.sep, subtext, self.sep))

    # Store warnings:
    self.warnings.append(subtext)

    # Print to screen and file:
    self.write(text)


  def error(self, message, tracklev=-2):
    """
    Pretty-print error message and end the code execution.

    Parameters
    ----------
    message: String
       String to be printed.
    tracklev: Integer
       Traceback level of error.
    """
    # Trace back the file, function, and line where the error source:
    trace = traceback.extract_stack()
    # Extract fields:
    modpath  = trace[tracklev][0]
    modname  = modpath[modpath.rfind('/')+1:]
    funcname = trace[tracklev][2]
    linenum  = trace[tracklev][1]

    # Generate string to print:
    subtext = self.msg(message, verb=1, indent=4, noprint=True)
    text = ("\n{:s}"
            "\n  Error in module: '{:s}', function: '{:s}', line: {:d}"
            "\n{:s}"
            "\n{:s}".
            format(self.sep, modname, funcname, linenum, subtext, self.sep))

    # Print to screen and file:
    self.write(text)
    # Close and exit:
    self.close()
    sys.exit(0)


  def progressbar(self, frac):
    """
    Print out to screen [and file] a progress bar, percentage,
    and current time.

    Parameters
    ----------
    frac: Float
       Fraction of the task that has been completed, ranging from 0.0 (none)
       to 1.0 (completed).
    """
    if self.verb < 1:
      return
    barlen = int(np.clip(round(10*frac), 0, 10))
    bar = ":"*barlen + " "*(10-barlen)

    text = "\n[{:s}] {:5.1f}% completed  ({:s})".format(bar, 100*frac,
                                                        time.ctime())
    # Print to screen and to file:
    self.write(text)


  def close(self):
    """
    Close log FILE pointer.
    """
    if self.file is not None:
      self.file.close()
