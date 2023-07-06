py"""
from pysmt.shortcuts import Symbol, LE, Plus, Real, Times, GE
from pysmt.typing import REAL
from pysmt.smtlib.commands import DECLARE_CONST, ASSERT
from pysmt.smtlib.script import SmtLibScript
import re
import math

from collections import namedtuple
import pysmt.smtlib.commands as smtcmd
from pysmt.exceptions import UnknownSmtLibCommandError
from pysmt.smtlib.printers import SmtPrinter, SmtDagPrinter, quote


def create_constraints(script, c, d):
    assert c.shape[0] == d.shape[0]
    num_constraints = c.shape[0]
    num_variables = c.shape[1]

    for i in range(num_constraints):
        times = []
        plus = {}
        for j in range(num_variables):
            if not (math.isclose(c[i][j], 0, rel_tol=1e-5)):
                x = Symbol(f"X_{j}", typename=REAL)
                times.append(Times(Real(float(c[i][j])), x))

        plus = Plus(times)
        le = LE(plus, Real(float(d[i])))
        declare_formula(script, le)


def calculate_fractions(path):
    s = ''
    with open(path, 'r') as file:
        for line in file:
            while len(re.findall(r'\(/ [0-9]+\s[0-9]+\)', line)) > 0:
                division = re.findall(r"\(/ [0-9]+\s[0-9]+\)", line)
                numbers = re.findall('[0-9]+', division.pop(0))
                number = float(numbers[0]) / float(numbers[1])
                line = re.sub(r"\(/ [0-9]+\s[0-9]+\)", str(number), line, 1)
            s += line

    with open(path, 'w') as file:
        file.write(s)


def get_output_constraints(path):
    x = False
    s = ""
    with open(path, 'r') as file:
        for line in file:
            if len(re.findall('Output constraints:', line)) == 1:
                x = True
            if x:
                s += line
    return s


def declare_const(script, formula):
    variables = formula.get_free_variables()
    for symbol in variables:
        script.commands.append(SmtLibCommand(name=DECLARE_CONST, args=[symbol]))


def declare_formula(script, formula):
    script.add_command(SmtLibCommand(name=ASSERT, args=[formula]))
    return script


def create_input_variables(script, num_inputs):
    for i in range(num_inputs):
        x = Symbol(f"X_{i}", typename=REAL)
        declare_const(script, x)


def create_output_variables(script, num_outputs):
    for i in range(num_outputs):
        y = Symbol(f"Y_{i}", typename=REAL)
        declare_const(script, y)


def create_vnnlib(c, d, num_inputs, num_outputs, input_filename, output_filename):
    
    script = SmtLibScript()
    create_input_variables(script, num_inputs)
    create_output_variables(script, num_outputs)
    create_constraints(script, c, d)
    script.to_file(output_filename, daggify=False)

    s = get_output_constraints(input_filename)
    with open(output_filename, 'a') as file:
        file.write(s)
    calculate_fractions(output_filename)


def create_vnnlib_from_lower_upper_bound(constraints, num_inputs, num_outputs, input_filename, output_filename):
    script = SmtLibScript()
    create_input_variables(script, num_inputs)
    create_output_variables(script, num_outputs)
    create_with_lower_upper_bounds(script, constraints)
    script.to_file(output_filename, daggify=False)
    c = get_constraints_as_string(constraints)
    s = get_output_constraints(input_filename)
    with open(output_filename, 'a') as file:
        file.write(c)
        file.write(s)
    calculate_fractions(output_filename)


def create_with_lower_upper_bounds(script, constraints):
    num_inputs = constraints.shape[0]
    for i in range(num_inputs):
        x = Symbol(f"X_{i}", typename=REAL)
        formula_le = GE(x, Real(float(constraints[i][0])))
        formula_ge = LE(x, Real(float(constraints[i][1])))
        declare_formula(script, formula_le)
        declare_formula(script, formula_ge)

def get_constraints_as_string(input_bounds):
    c = ""

    for i in range(input_bounds.shape[0]):
        c += f"(assert (<= X_{i} {input_bounds[i, 1]}))\n"
        c += f"(assert (>= X_{i} {input_bounds[i, 0]}))\n"
        c += "\n"
    c += "\n"
    return c

class SmtLibCommand(namedtuple('SmtLibCommand', ['name', 'args'])):
    def serialize(self, outstream=None, printer=None, daggify=True):

        if (outstream is None) and (printer is not None):
            outstream = printer.stream
        elif (outstream is not None) and (printer is None):
            if daggify:
                printer = SmtDagPrinter(outstream)
            else:
                printer = SmtPrinter(outstream)
        else:
            assert (outstream is not None and printer is not None) or \
                   (outstream is None and printer is None), \
                   "Exactly one of outstream and printer must be set."

        if self.name == smtcmd.SET_OPTION:
            outstream.write("(%s %s %s)" % (self.name, self.args[0], self.args[1]))

        elif self.name == smtcmd.SET_INFO:
            outstream.write("(%s %s %s)" % (self.name, self.args[0],
                                            quote(self.args[1])))

        elif self.name == smtcmd.ASSERT:
            outstream.write("(%s " % self.name)
            printer.printer(self.args[0])
            outstream.write(")")

        elif self.name == smtcmd.GET_VALUE:
            outstream.write("(%s (" % self.name)
            for a in self.args:
                printer.printer(a)
                outstream.write(" ")
            outstream.write("))")

        elif self.name in [smtcmd.CHECK_SAT, smtcmd.EXIT,
                           smtcmd.RESET_ASSERTIONS, smtcmd.GET_UNSAT_CORE,
                           smtcmd.GET_ASSIGNMENT, smtcmd.GET_MODEL]:
            outstream.write("(%s)" % self.name)

        elif self.name == smtcmd.SET_LOGIC:
            outstream.write("(%s %s)" % (self.name, self.args[0]))

        elif self.name in [smtcmd.DECLARE_FUN, smtcmd.DECLARE_CONST]:
            symbol = self.args[0]
            type_str = symbol.symbol_type()
            outstream.write("(%s %s %s)" % (self.name,
                                            quote(symbol.symbol_name()),
                                            type_str))

        elif self.name == smtcmd.DEFINE_FUN:
            name = self.args[0]
            params_list = self.args[1]
            params = " ".join(["(%s %s)" % (v, v.symbol_type().as_smtlib(funstyle=False)) for v in params_list])
            rtype = self.args[2]
            expr = self.args[3]
            outstream.write("(%s %s (%s) %s " % (self.name, name, params, rtype.as_smtlib(funstyle=False)))
            printer.printer(expr)
            outstream.write(")")

        elif self.name in [smtcmd.PUSH, smtcmd.POP]:
            outstream.write("(%s %d)" % (self.name, self.args[0]))

        elif self.name == smtcmd.DEFINE_SORT:
            name = self.args[0]
            params_list = self.args[1]
            params = " ".join(x.as_smtlib(funstyle=False) for x in params_list)
            rtype = self.args[2]
            outstream.write("(%s %s (%s) %s)" % (self.name,
                                                 name,
                                                 params,
                                                 rtype.as_smtlib(funstyle=False)))
        elif self.name == smtcmd.DECLARE_SORT:
            type_decl = self.args[0]
            outstream.write("(%s %s %d)" % (self.name,
                                            type_decl.name,
                                            type_decl.arity))

        elif self.name in smtcmd.ALL_COMMANDS:
            raise NotImplementedError("'%s' is a valid SMT-LIB command "
                                      "but it is currently not supported. "
                                      "Please open a bug-report." % self.name)
        else:
            raise UnknownSmtLibCommandError(self.name)

"""

global create_vnnlib = py"create_vnnlib"
global create_vnnlib_from_lower_upper_bound = py"create_vnnlib_from_lower_upper_bound"
