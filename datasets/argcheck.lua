-- This file is borrowed from https://github.com/fmassa/object-detection.torch

local usage = require 'argcheck.usage'
local env = require 'argcheck.env'
--------------------------------------------------------------------------------
-- Simple argument function with a similar interface to argcheck, but which
-- supports lots of default arguments for named rules.
-- Not as fast and elegant though.
--------------------------------------------------------------------------------
local function argcheck(rules)
  -- basic checks
  assert(not (rules.noordered and rules.nonamed), 'rules must be at least ordered or named')
  assert(rules.help == nil or type(rules.help) == 'string', 'rules help must be a string or nil')
  assert(rules.doc == nil or type(rules.doc) == 'string', 'rules doc must be a string or nil')
  assert(not rules.overload, 'rules overload not supported')
  assert(not (rules.doc and rules.help), 'choose between doc or help, not both')
  for _, rule in ipairs(rules) do
    assert(rule.name, 'rule must have a name field')
    assert(rule.type == nil or type(rule.type) == 'string', 'rule type must be a string or nil')
    assert(rule.help == nil or type(rule.help) == 'string', 'rule help must be a string or nil')
    assert(rule.doc == nil or type(rule.doc) == 'string', 'rule doc must be a string or nil')
    assert(rule.check == nil or type(rule.check) == 'function', 'rule check must be a function or nil')
    --assert(rule.defaulta == nil or type(rule.defaulta) == 'string', 'rule defaulta must be a string or nil')
    --assert(rule.defaultf == nil or type(rule.defaultf) == 'function', 'rule defaultf must be a function or nil')
  end

  if not (rules.pack == nil or rules.pack) then
    error('pack need to be true')
  end
  if rules.nonamed then
    error('only named arguments')
  end

  local arginfo = {}
  for k,v in ipairs(rules) do
    arginfo[v.name] = k
  end

  local function func(args)

    local iargs = {}
    for _,rule in ipairs(rules) do
      iargs[rule.name] = rule.default
      if rule.default == nil and 
        args[rule.name] == nil and 
        rule.opt ~= true then
        print(usage(rules))
        error('Missing argument: '..rule.name)
      end
    end

    for k,v in pairs(args) do
      if not env.istype(v,rules[arginfo[k]].type) then
        print(usage(rules))
        error('Wrong type: '.. k)
      end

      if rules[arginfo[k]].check then
        local c = rules[arginfo[k]].check(args[k])
        if not c then
          print(usage(rules))
          error('check did not pass')
        end
      end
      iargs[k] = args[k]
    end

    return iargs
  end

  return func

end

return argcheck
