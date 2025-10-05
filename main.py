# Copyright (c) 2025 Can Joshua Lehmann

class Expr:
    def __init__(self, args):
        self.args = [*args]

    def collect(self, type, into=None):
        if into is None:
            into = set()
        if isinstance(self, type):
            into.add(self)
        for arg in self.args:
            arg.collect(type, into)
        return into
    
    def _subst_recur(self, fn):
        raise NotImplementedError()

    def subst(self, fn):
        sub = fn(self)
        if sub is not None:
            return sub
        else:
            return self._subst_recur(fn)

class Const(Expr):
    def __init__(self, value):
        super().__init__([])
        self.value = value
    
    def _subst_recur(self, fn):
        return self

    def format(self, level):
        return str(self.value)

class Var(Expr):
    def __init__(self, name):
        super().__init__([])
        self.name = name
    
    def _subst_recur(self, fn):
        return self
    
    def format(self, level):
        return self.name

class Sum(Expr):
    def __init__(self, *args):
        super().__init__(args)

    def simpl(*args):
        const = 0
        terms = []
        for arg in args:
            match arg:
                case Const(value=value):
                    const += value
                case Sum(args=nested):
                    terms += nested
                case _:
                    terms.append(arg)
        if const != 0:
            terms.append(Const(const))
        match terms:
            case []: return Const(0)
            case [x]: return x
            case _: return Sum(*terms)

    def _subst_recur(self, fn):
        return Sum.simpl(*(arg.subst(fn) for arg in self.args))

    def format(self, level):
        res = ""
        for i, arg in enumerate(self.args):
            if i > 0:
                res += " + "
            res += arg.format(30)
        if level > 30:
            res = "(" + res + ")"
        return res

class Prod(Expr):
    def __init__(self, *args):
        super().__init__(args)

    def simpl(*args):
        const = 1
        terms = []
        for arg in args:
            match arg:
                case Const(value=value):
                    const *= value
                case Prod(args=nested):
                    terms += nested
                case _:
                    terms.append(arg)
        if const == 0:
            return Const(0)
        if const != 1:
            terms.append(Const(const))
        match terms:
            case []: return Const(1)
            case [x]: return x
            case _: return Prod(*terms)
    
    def _subst_recur(self, fn):
        return Prod.simpl(*(arg.subst(fn) for arg in self.args))

    def format(self, level):
        res = ""
        for i, arg in enumerate(self.args):
            if i > 0:
                res += " * "
            res += arg.format(40)
        if level > 40:
            res = "(" + res + ")"
        return res

class Neg(Expr):
    def __init__(self, arg):
        super().__init__([arg])

    def simpl(arg):
        match arg:
            case Const(value=value):
                return Const(-value)
            case Neg(args=[x]):
                return x
            case _:
                return Neg(arg)

    def _subst_recur(self, fn):
        return Neg.simpl(self.args[0].subst(fn))

    def format(self, level):
        res = "-" + self.args[0].format(50)
        if level > 50:
            res = "(" + res + ")"
        return res

class Inv(Expr):
    def __init__(self, arg):
        super().__init__([arg])
    
    def simpl(arg):
        match arg:
            case Const(value=value):
                return Const(1 / value)
            case Inv(args=[x]):
                return x
            case _:
                return Inv(arg)

    def _subst_recur(self, fn):
        return Inv.simpl(self.args[0].subst(fn))

    def format(self, level):
        res = "1 / " + self.args[0].format(60)
        if level > 60:
            res = "(" + res + ")"
        return res

class Dt(Expr):
    def __init__(self, arg):
        super().__init__([arg])
    
    def _subst_recur(self, fn):
        return Dt(self.args[0].subst(fn))

    def format(self, level):
        res = self.args[0].format(70) + "'"
        if level >= 70:
            res = "(" + res + ")"
        return res

class Sym:
    def __init__(self, expr):
        if isinstance(expr, str):
            self.expr = Var(expr)
        else:
            self.expr = self._to_expr(expr)

    def _to_expr(self, value):
        match value:
            case int() | float():
                return Const(value)
            case Expr():
                return value
            case Sym(expr=expr):
                return expr
            case _:
                raise TypeError(f"Cannot convert {value} to expression")
    def __add__(self, other):
        return Sym(Sum.simpl(self.expr, self._to_expr(other)))
    
    def __radd__(self, other):
        return Sym(Sum.simpl(self._to_expr(other), self.expr))
    
    def __sub__(self, other):
        return Sym(Sum.simpl(self.expr, Neg.simpl(self._to_expr(other))))
    
    def __rsub__(self, other):
        return Sym(Sum.simpl(self._to_expr(other), Neg.simpl(self.expr)))
    
    def __mul__(self, other):
        return Sym(Prod.simpl(self.expr, self._to_expr(other)))
    
    def __rmul__(self, other):
        return Sym(Prod.simpl(self._to_expr(other), self.expr))
    
    def __truediv__(self, other):
        return Sym(Prod.simpl(self.expr, Inv.simpl(self._to_expr(other))))
    
    def __rtruediv__(self, other):
        return Sym(Prod.simpl(self._to_expr(other), Inv.simpl(self.expr)))
    
    def __neg__(self):
        return Sym(Neg.simpl(self.expr))
    
    def __invert__(self):
        return Sym(Inv.simpl(self.expr))
    
    def dt(self):
        return Sym(Dt(self.expr))

    def __str__(self):
        return self.expr.format(0)
    
    def __repr__(self):
        return f"Sym({self.expr.format(0)})"

class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
    
    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while item != root:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return item
    
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a

class ODESystem:
    def __init__(self, eqs, params):
        self.eqs = []
        self.state = {}
        self.params = self._params_to_names(params)

        eqs = self._eqs_to_exprs(eqs)
        self._create_dt_vars(eqs)

    def _eqs_to_exprs(self, eqs):
        exprs = []
        for eq in eqs:
            if isinstance(eq, Sym):
                eq = eq.expr
            assert isinstance(eq, Expr)
            exprs.append(eq)
        return exprs
    
    def _params_to_names(self, params):
        names = set()
        for param in params:
            if isinstance(param, Sym):
                param = param.expr
            if isinstance(param, Var):
                param = param.name
            assert isinstance(param, str)
            names.add(param)
        return names

    def _create_dt_vars(self, eqs):
        dt_vars = set()
        for eq in eqs:
            for dt in eq.collect(Dt):
                assert isinstance(dt.args[0], Var)
                var = dt.args[0].name
                dt_vars.add(var)
        
        for var in dt_vars:
            self.state[var] = "__dt_" + var

        def subst(expr):
            if isinstance(expr, Dt):
                assert isinstance(expr.args[0], Var)
                var = expr.args[0].name
                dt_vars.add(var)
                return Var(self.state[var])
            return None

        for eq in eqs:
            self.eqs.append(eq.subst(subst))

    def __str__(self):
        return "\n".join(f"0 = {eq.format(0)}" for eq in self.eqs)

def remove_aliasing(ode_system):
    vars = set()
    for eq in ode_system.eqs:
        vars |= eq.collect(Var)

    alises = UnionFind([var.name for var in vars])
    eqs = []
    for eq in ode_system.eqs:
        match eq:
            case Sum(args=[Var(name=lhs_name), Neg(args=[Var(name=rhs_name)])]):
                alises.union(lhs_name, rhs_name)
            case Sum(args=[Neg(args=[Var(name=lhs_name)]), Var(name=rhs_name)]):
                alises.union(lhs_name, rhs_name)
            case _:
                eqs.append(eq)
    
    ode_system.eqs = [
        eq.subst(lambda e: Var(alises.find(e.name)) if isinstance(e, Var) else None)
        for eq in eqs
    ]


def simplify(ode_system):
    remove_aliasing(ode_system)
    
class Action:
    def __init__(self):
        pass

class EvalAction(Action):
    def __init__(self, var, expr):
        super().__init__()
        self.var = var
        self.expr = expr
    
    def __repr__(self):
        return f"EvalAction({self.var} = {self.expr.format(0)})"

class SolveAction(Action):
    def __init__(self, eqs):
        super().__init__()
        self.eqs = eqs

    def __repr__(self):
        return "SolveAction([" + ", ".join(eq.format(0) for eq in self.eqs) + "])"

def rearrange(eq, solve_for):
    def count_occurrences(expr, var):
        if isinstance(expr, Var):
            return int(expr.name == var)
        return sum(count_occurrences(arg, var) for arg in expr.args)
    
    def find_path(expr, var):
        if isinstance(expr, Var) and expr.name == var:
            return []
        for i, arg in enumerate(expr.args):
            path = find_path(arg, var)
            if path is not None:
                return [i] + path
        return None

    if count_occurrences(eq, solve_for) == 1:
        path = find_path(eq, solve_for)
        if path is not None:
            rhs = Const(0)
            lhs = eq
            for index in path:
                match lhs:
                    case Sum(args=args):
                        for i, arg in enumerate(args):
                            if i != index:
                                rhs = Sum.simpl(rhs, Neg.simpl(arg))
                    case Prod(args=args):
                        for i, arg in enumerate(args):
                            if i != index:
                                rhs = Prod.simpl(rhs, Inv.simpl(arg))
                    case Neg(args=[arg]):
                        rhs = Neg.simpl(rhs)
                    case Inv(args=[arg]):
                        rhs = Inv.simpl(rhs)
                    case _:
                        return None
                lhs = lhs.args[index]

            assert isinstance(lhs, Var) and lhs.name == solve_for
            return rhs

    return None

def schedule(ode_system):
    known = set()
    for var, dt_var in ode_system.state.items():
        known.add(var)
    known |= ode_system.params
    
    schedule = []

    eqs = [*ode_system.eqs]
    eq_vars = {eq: {var.name for var in eq.collect(Var)} for eq in eqs}
    changed = True
    while changed:
        changed = False
        for i, eq in enumerate(eqs):
            vars = eq_vars[eq]
            if len(vars - known) == 1:
                solve_for = (vars - known).pop()
                expr = rearrange(eq, solve_for)
                if expr is not None:
                    schedule.append(EvalAction(solve_for, expr))
                    known.add(solve_for)
                    eqs.remove(eq)
                    changed = True
                    break

    if eqs:
        schedule.append(SolveAction(eqs))

    return schedule



if __name__ == "__main__":
    # Lotka-Volterra equations
    x = Sym("x")
    y = Sym("y")
    a = Sym("a")
    b = Sym("b")
    c = Sym("c")
    d = Sym("d")
    temp = Sym("temp")
    temp2 = Sym("temp2")
    eqs = [
        temp - x * y,
        temp - temp2,
        x.dt() - a * x - b * temp2,
        y.dt() - d * temp - c * y
    ]

    ode_system = ODESystem(eqs, params=[a, b, c, d])
    print(ode_system)
    simplify(ode_system)
    print(ode_system)
    actions = schedule(ode_system)
    print(actions)