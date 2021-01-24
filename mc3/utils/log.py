# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
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
  def __init__(self, logname=None, verb=2, append=False, width=70):
      """
      Parameters
      ----------
      logname: String
          Name of FILE pointer where to store log entries. Set to None to
          print only to stdout.
      verb: Integer
          Conditional threshold to print messages.  There are five levels
          of increasing verbosity:
          verb <  0: only print error() calls.
          verb >= 0: print warning() calls.
          verb >= 1: print head() calls.
          verb >= 2: print msg() calls.
          verb >= 3: print debug() calls.
      append: Bool
          If True, append logged text to existing file.
          If False, write logs to new file.
      width: Integer
          Maximum length of each line of text (longer texts will be break
          down into multiple lines).
      """
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

  def __enter__(self):
      return self

  def __exit__(self, type, value, traceback):
      self.close()


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
      if self.file is not None and not self.file.closed:
          self.file.write(text + "\n")
          self.file.flush()


  def wrap(self, message, indent=None, si=None, width=None):
      """
      Wrap text according to given/default indentation and width.

      Parameters
      ----------
      message: String
          String to be printed.
      indent: Integer
          Number of blank spaces to indent the printed message.
      si: Integer
          Sub-sequent-lines indentation.
      width: Integer
          If not None, override text width (only for this specific call).

      Returns
      -------
      text: String
          Formatted output string.
      """
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
      msg = [textwrap.fill(
                 sentence,
                 break_long_words=False,
                 break_on_hyphens=False,
                 initial_indent=indspace,
                 subsequent_indent=sind,
                 width=width) for sentence in message.splitlines()]

      return "\n".join(msg)


  def debug(self, message, indent=None, si=None, width=None):
      """
      Print wrapped message to screen and file if verbosity is > 2.

      Parameters
      ----------
      message: String
          String to be printed.
      indent: Integer
          Number of blank spaces to indent the printed message.
      si: Integer
          Sub-sequent-lines indentation.
      width: Integer
          If not None, override text width (only for this specific call).
      """
      if self.verb < 3:
          return
      text = self.wrap(message, indent, si, width)
      self.write(text)


  def msg(self, message, indent=None, si=None, width=None):
      """
      Print wrapped message to screen and file if verbosity is > 1.

      Parameters
      ----------
      message: String
          String to be printed.
      indent: Integer
          Number of blank spaces to indent the printed message.
      si: Integer
          Sub-sequent-lines indentation.
      width: Integer
          If not None, override text width (only for this specific call).
      """
      if self.verb < 2:
          return
      text = self.wrap(message, indent, si, width)
      self.write(text)


  def head(self, message, indent=None, si=None, width=None):
      """
      Print wrapped message to screen and file if verbosity is > 0.

      Parameters
      ----------
      message: String
          String to be printed.
      indent: Integer
          Number of blank spaces to indent the printed message.
      si: Integer
          Sub-sequent-lines indentation.
      width: Integer
          If not None, override text width (only for this specific call).
      """
      if self.verb < 1:
          return
      text = self.wrap(message, indent, si, width)
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
      subtext = self.wrap(message, indent=4)
      # Add the warning surroundings:
      text = (
          f"\n{self.sep}"
           "\n  Warning:"
          f"\n{subtext}"
          f"\n{self.sep}\n")

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
      subtext = self.wrap(message, indent=4)
      text = (
          f"\n{self.sep}"
          f"\n  Error in module: '{modname}', function: '{funcname}', "
          f"line: {linenum}"
          f"\n{subtext}"
          f"\n{self.sep}")

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
          Fraction of the task that has been completed, ranging from
          0.0 (none) to 1.0 (completed).
      """
      if self.verb < 1:
          return
      barlen = int(np.clip(round(10*frac), 0, 10))
      bar = ":"*barlen + " "*(10-barlen)

      text = f"\n[{bar}] {100*frac:5.1f}% completed  ({time.ctime()})"
      # Print to screen and to file:
      self.write(text)


  def close(self):
      """
      Close log FILE pointer.
      """
      if self.file is not None:
          self.file.close()
