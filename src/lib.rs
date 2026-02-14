use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

fn perf_counter_ns(py: Python<'_>) -> PyResult<u64> {
    // PyO3 0.22 uses bound imports.
    let time_mod = py.import_bound("time")?;
    time_mod
        .getattr("perf_counter_ns")?
        .call0()?
        .extract::<u64>()
}

#[derive(Clone, Debug, Eq)]
enum Key {
    Int(i64),
    Str(String),
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Key::Int(a), Key::Int(b)) => a == b,
            (Key::Str(a), Key::Str(b)) => a == b,
            _ => false,
        }
    }
}

impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Key::Int(i) => {
                0u8.hash(state);
                i.hash(state);
            }
            Key::Str(s) => {
                1u8.hash(state);
                s.hash(state);
            }
        }
    }
}

fn key_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Key> {
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(Key::Int(v));
    }
    if let Ok(s) = obj.downcast::<PyString>() {
        return Ok(Key::Str(s.to_str()?.to_owned()));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "key must be int or str",
    ))
}

#[pyclass(frozen, module = "timekid._fast")]
#[derive(Clone)]
struct Token {
    key: Key,
    t0_ns: u64,
}

#[pymethods]
impl Token {
    #[getter]
    fn t0_ns(&self) -> u64 {
        self.t0_ns
    }
}

#[pyclass(module = "timekid._fast")]
struct FastTimer {
    key_map: HashMap<String, i64>,
    next_key: i64,
    times_ns: HashMap<Key, Vec<u64>>,
}

#[pymethods]
impl FastTimer {
    #[new]
    fn new() -> Self {
        Self {
            key_map: HashMap::new(),
            next_key: 0,
            times_ns: HashMap::new(),
        }
    }

    fn key_id(&mut self, name: &str) -> i64 {
        if let Some(k) = self.key_map.get(name) {
            return *k;
        }
        let kid = self.next_key;
        self.next_key += 1;
        self.key_map.insert(name.to_owned(), kid);
        kid
    }

    fn start(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Token> {
        let k = key_from_py(key)?;
        let t0_ns = perf_counter_ns(py)?;
        Ok(Token { key: k, t0_ns })
    }

    fn stop(&mut self, py: Python<'_>, token: &Token) -> PyResult<u64> {
        let t1_ns = perf_counter_ns(py)?;
        let dt = t1_ns.saturating_sub(token.t0_ns);
        self.times_ns.entry(token.key.clone()).or_default().push(dt);
        Ok(dt)
    }

    #[pyo3(signature = (key=None))]
    fn clear(&mut self, key: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        match key {
            None => {
                self.times_ns.clear();
                Ok(())
            }
            Some(obj) => {
                let k = key_from_py(obj)?;
                self.times_ns.remove(&k);
                Ok(())
            }
        }
    }

    #[getter]
    fn times_ns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (k, v) in self.times_ns.iter() {
            match k {
                Key::Int(i) => d.set_item(*i, v.clone())?,
                Key::Str(s) => d.set_item(s, v.clone())?,
            }
        }
        Ok(d)
    }

    #[pyo3(signature = (precision=None))]
    fn times_s<'py>(&self, py: Python<'py>, precision: Option<u32>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (k, arr) in self.times_ns.iter() {
            let mut vals: Vec<f64> = arr.iter().map(|ns| (*ns as f64) / 1e9).collect();
            if let Some(p) = precision {
                let factor = 10_f64.powi(p as i32);
                for x in vals.iter_mut() {
                    *x = (*x * factor).round() / factor;
                }
            }
            match k {
                Key::Int(i) => d.set_item(*i, vals)?,
                Key::Str(s) => d.set_item(s, vals)?,
            }
        }
        Ok(d)
    }
}

#[pymodule]
fn _fast(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastTimer>()?;
    m.add_class::<Token>()?;
    Ok(())
}
