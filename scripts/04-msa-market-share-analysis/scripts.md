# 4. SCRIPTS MSA (MARKET SHARE ANALYSIS)

**Nota:** *Los siguientes scripts están diseñados para el análisis de datos orientado a la estimación de market share, la generación de visualizaciones analíticas y la identificación de insights estratégicos a partir de datos históricos obtenidos de TradeMap. Se sugiere implementar este flujo de trabajo en entornos de ejecución Python tales como:*

- [Google Colab](https://colab.research.google.com/)
- [Python-Fiddle](https://python-fiddle.com/)
- [Kaggle Notebooks](https://www.kaggle.com/code)
- [Deepnote](https://deepnote.com/)

## 4.1. Importar e instalar librerías

```
# Instalar librerías si es necesario
!pip install openpyxl

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

## 4.2. Cargar archivos fuente (TradeMap) - Identificación flexible de columnas
```
import io
import re
import unicodedata

import pandas as pd

try:
    from google.colab import files
    EN_COLAB = True
except ImportError:
    EN_COLAB = False

# ------------------------------------------------------------
# CONFIGURACIÓN OPCIONAL (asignación manual por configuración)
# Dejar en None para usar detección automática.
#   COLUMNA_PAIS_MANUAL   -> nombre o índice. Ej: 'Exportadores' o 0
#   COLUMNAS_VALOR_MANUAL -> nombres, índices o rango. Ej: ['2020','2021'], [2,3,4] o '2-6'
# ------------------------------------------------------------
COLUMNA_PAIS_MANUAL = None
COLUMNAS_VALOR_MANUAL = None

# ------------------------------------------------------------
# Palabras clave (se comparan sin tildes y en minúsculas)
# ------------------------------------------------------------
KW_PAIS = ['exportadores', 'importadores', 'exporters', 'importers', 'partner', 'socio',
           'pais', 'country', 'proveedor', 'supplier', 'mercado', 'market',
           'origen', 'origin', 'destino', 'destination']
KW_VALOR = ['valor', 'value', 'importado', 'exportado', 'imported', 'exported', 'trade',
            'importaciones', 'exportaciones', 'imports', 'exports', 'usd']
KW_EXCLUIR = ['cantidad', 'quantity', 'unidad', 'unit', 'crecimiento', 'growth', 'participacion',
              'share', 'tasa', 'rate', 'arancel', 'tariff', 'peso', 'weight', 'ranking',
              'rank', 'saldo', 'balance', 'distancia', 'distance', 'concentracion', 'concentration']
NOMBRES_MUNDO = {'mundo', 'world', 'monde', 'total'}
EXT_SOPORTADAS = ('.xlsx', '.xlsm', '.xls', '.html', '.htm', '.csv', '.txt')


# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------
def _norm(s):
    s = unicodedata.normalize('NFKD', str(s))
    return ''.join(c for c in s if not unicodedata.combining(c)).lower().strip()


def _tiene_kw(texto, kws):
    t = _norm(texto)
    return any(k in t for k in kws)


def _anios_en(texto):
    return sorted(set(re.findall(r'(?<!\d)((?:19|20)\d{2})(?!\d)', str(texto))))


def _a_numero(v):
    """Convierte un valor a número admitiendo separadores de miles ('1,234,567' / '1.234.567')."""
    if pd.isna(v):
        return float('nan')
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace('\xa0', '').replace(' ', '')
    if re.fullmatch(r'-?\d{1,3}([.,]\d{3})+', s):            # solo miles
        s = re.sub(r'[.,]', '', s)
    elif ',' in s and '.' in s:                              # ambos: el último es decimal
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:                                           # solo coma decimal
        s = s.replace(',', '.')
    return pd.to_numeric(s, errors='coerce')


def _serie_numerica(serie):
    return serie.map(_a_numero).astype(float)


def _pct_numerico(serie):
    s = serie.dropna()
    if s.empty:
        return 0.0
    return float(_serie_numerica(s).notna().mean())


def _pct_texto(serie):
    s = serie.dropna()
    if s.empty:
        return 0.0
    num = _serie_numerica(s).notna()
    return float((~num).mean())


# ------------------------------------------------------------
# Lectura del archivo (.xlsx, .xls, .xls en HTML/texto)
# ------------------------------------------------------------
def leer_archivo_crudo(filename):
    """Devuelve un DataFrame sin cabecera (header=None) con todo el contenido de la hoja."""
    ext = '.' + filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
    if ext not in EXT_SOPORTADAS:
        raise ValueError(f"Formato '{ext}' no soportado. Use: {', '.join(EXT_SOPORTADAS)}")

    errores = []
    if ext in ('.xlsx', '.xlsm'):
        return pd.read_excel(filename, sheet_name=0, header=None, engine='openpyxl')

    if ext == '.xls':
        try:
            return pd.read_excel(filename, sheet_name=0, header=None, engine='xlrd')
        except Exception as e:
            errores.append(f"xlrd: {e}")
        # Muchos .xls de TradeMap son HTML o texto con otra extensión
        try:
            return pd.read_excel(filename, sheet_name=0, header=None, engine='openpyxl')
        except Exception as e:
            errores.append(f"openpyxl: {e}")

    if ext in ('.xls', '.html', '.htm'):
        try:
            with open(filename, 'rb') as f:
                contenido = f.read()
            texto = contenido.decode('utf-8', errors='replace')
            tablas = pd.read_html(io.StringIO(texto), header=None)
            if tablas:
                t = max(tablas, key=lambda t: t.shape[0] * t.shape[1])
                if not isinstance(t.columns, pd.RangeIndex):      # <th> se interpretó como cabecera: devolverla como fila
                    cab = pd.DataFrame([[c[-1] if isinstance(c, tuple) else c for c in t.columns]])
                    t = pd.concat([cab, t.set_axis(range(t.shape[1]), axis=1)], ignore_index=True)
                return t.reset_index(drop=True)
        except Exception as e:
            errores.append(f"html: {e}")

    if ext in ('.xls', '.csv', '.txt'):
        for enc in ('utf-8', 'latin-1', 'utf-16'):
            try:
                return pd.read_csv(filename, sep=None, engine='python', header=None, encoding=enc)
            except Exception as e:
                errores.append(f"texto({enc}): {e}")

    raise ValueError("No fue posible leer el archivo. Detalle:\n  - " + "\n  - ".join(errores))


def detectar_fila_cabecera(raw, max_filas=30):
    """Detecta la fila de cabecera aunque existan filas de título antes de la tabla."""
    mejor, mejor_score = 0, -1
    for i in range(min(max_filas, len(raw))):
        fila = raw.iloc[i]
        celdas = [c for c in fila.tolist() if pd.notna(c) and str(c).strip() != '']
        if len(celdas) < 2:
            continue
        score = len(celdas)
        score += 5 * any(_tiene_kw(c, KW_PAIS) for c in celdas)
        score += 3 * sum(1 for c in celdas if _anios_en(c))
        score -= 2 * sum(1 for c in celdas if isinstance(c, (int, float)) and not _anios_en(c))
        if score > mejor_score:
            mejor, mejor_score = i, score
    return mejor


def construir_dataframe(raw):
    """Aplica la fila de cabecera detectada y limpia nombres duplicados / vacíos."""
    h = detectar_fila_cabecera(raw)
    nombres, vistos = [], {}
    for j, c in enumerate(raw.iloc[h].tolist()):
        if isinstance(c, float) and pd.notna(c) and c == int(c):
            c = int(c)                                       # 2020.0 -> 2020
        n = f"columna_{j}" if pd.isna(c) or str(c).strip() == '' else str(c).strip()
        if n in vistos:
            vistos[n] += 1
            n = f"{n}_{vistos[n]}"
        else:
            vistos[n] = 0
        nombres.append(n)
    df = raw.iloc[h + 1:].copy()
    df.columns = nombres
    df = df.dropna(how='all').reset_index(drop=True)
    return df, h


# ------------------------------------------------------------
# Detección automática
# ------------------------------------------------------------
def detectar_columnas_valor(df):
    cand = []
    for j, col in enumerate(df.columns):
        anios = _anios_en(col)
        if len(anios) != 1:                                  # sin año o rango de años (crecimiento)
            continue
        if _tiene_kw(col, KW_EXCLUIR):
            continue
        if _pct_numerico(df[col]) < 0.5:
            continue
        cand.append(j)
    # Si hay varias familias (ej. "Valor importado en 2020" y otra con año), priorizar las que traen palabra clave de valor
    con_kw = [j for j in cand if _tiene_kw(df.columns[j], KW_VALOR)]
    if con_kw and len(con_kw) >= 1:
        return con_kw
    return cand


def detectar_columna_pais(df, idx_valor):
    resto = [j for j in range(df.shape[1]) if j not in idx_valor]
    por_kw = [j for j in resto if _tiene_kw(df.columns[j], KW_PAIS) and _pct_texto(df.iloc[:, j]) >= 0.5]
    if len(por_kw) == 1:
        return por_kw[0]
    if len(por_kw) > 1:
        return max(por_kw, key=lambda j: _pct_texto(df.iloc[:, j]))
    por_contenido = [j for j in resto if _pct_texto(df.iloc[:, j]) >= 0.8]
    if len(por_contenido) == 1:
        return por_contenido[0]
    return None                                              # no concluyente


# ------------------------------------------------------------
# Interfaz de selección y asignación manual
# ------------------------------------------------------------
def mostrar_columnas(df):
    filas = []
    for j, col in enumerate(df.columns):
        ejemplo = df.iloc[:, j].dropna()
        filas.append({
            'indice': j,
            'nombre': str(col),
            'ejemplo': str(ejemplo.iloc[0])[:30] if len(ejemplo) else '',
            '% numérico': f"{_pct_numerico(df.iloc[:, j]) * 100:.0f}%",
        })
    print("\nColumnas disponibles:")
    print(pd.DataFrame(filas).to_string(index=False))


def _resolver_una(spec, df):
    """Resuelve un nombre o índice a posición de columna."""
    if isinstance(spec, (int,)) or (isinstance(spec, str) and spec.strip().lstrip('-').isdigit()
                                    and str(spec).strip() not in map(str, df.columns)):
        j = int(spec)
        if not 0 <= j < df.shape[1]:
            raise ValueError(f"Índice {j} fuera de rango (0-{df.shape[1] - 1}).")
        return j
    s = str(spec).strip()
    for j, c in enumerate(df.columns):
        if str(c).strip() == s:
            return j
    for j, c in enumerate(df.columns):
        if _norm(c) == _norm(s):
            return j
    raise ValueError(f"La columna '{spec}' no existe en el archivo.")


def parsear_seleccion(spec, df):
    """Acepta: nombre, índice, lista, 'a-b' (rango) o 'a,b,c'."""
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        res = []
        for s in spec:
            res.extend(parsear_seleccion(s, df))
        return res
    if isinstance(spec, str):
        res = []
        for parte in [p.strip() for p in spec.split(',') if p.strip()]:
            m = re.fullmatch(r'(\d+)\s*-\s*(\d+)', parte)
            if m and parte not in map(str, df.columns):
                a, b = int(m.group(1)), int(m.group(2))
                res.extend(range(a, b + 1))
            else:
                res.append(_resolver_una(parte, df))
        return res
    return [_resolver_una(spec, df)]


def seleccion_interactiva(df, pais_idx, valor_idx):
    mostrar_columnas(df)
    if pais_idx is None:
        entrada = input("\nColumna de PAÍSES (nombre o índice): ")
        pais_idx = parsear_seleccion(entrada, df)[0]
    if not valor_idx:
        entrada = input("Columnas de VALOR (índices/nombres separados por coma o rango 'a-b'): ")
        valor_idx = parsear_seleccion(entrada, df)
    return pais_idx, valor_idx


# ------------------------------------------------------------
# Validación
# ------------------------------------------------------------
def validar_seleccion(df, pais_idx, valor_idx):
    errores = []
    if pais_idx is None:
        errores.append("No se identificó la columna de países.")
    elif _pct_texto(df.iloc[:, pais_idx]) < 0.5:
        errores.append(f"La columna de países '{df.columns[pais_idx]}' es mayormente numérica.")
    if not valor_idx:
        errores.append("No se identificó ninguna columna de valores comerciales.")
    if pais_idx is not None and pais_idx in valor_idx:
        errores.append("La columna de países también fue seleccionada como columna de valores.")
    anios_vistos = {}
    for j in valor_idx:
        col = df.columns[j]
        anios = _anios_en(col)
        if len(anios) != 1:
            errores.append(f"La columna de valores '{col}' no tiene un único año en su cabecera.")
        else:
            if anios[0] in anios_vistos:
                errores.append(f"Año {anios[0]} repetido: '{anios_vistos[anios[0]]}' y '{col}'.")
            anios_vistos[anios[0]] = col
        if _pct_numerico(df.iloc[:, j]) == 0:
            errores.append(f"La columna de valores '{col}' no contiene números.")
    return errores


# ------------------------------------------------------------
# Proceso principal
# ------------------------------------------------------------
def procesar_trademap(filename, pais_manual=None, valores_manual=None, interactivo=True):
    raw = leer_archivo_crudo(filename)
    df_raw, fila_cab = construir_dataframe(raw)

    print(f"Fila de cabecera detectada: {fila_cab}")
    print("Columnas disponibles:", df_raw.columns.tolist())
    print("\nPrimeras 5 filas:")
    print(df_raw.head())

    # 1) Asignación manual por configuración (tiene prioridad)
    pais_idx, valor_idx = None, []
    if pais_manual is not None:
        pais_idx = parsear_seleccion(pais_manual, df_raw)[0]
    if valores_manual is not None:
        valor_idx = parsear_seleccion(valores_manual, df_raw)

    # 2) Detección automática para lo que falte
    if not valor_idx:
        valor_idx = detectar_columnas_valor(df_raw)
    if pais_idx is None:
        pais_idx = detectar_columna_pais(df_raw, valor_idx)

    # 3) Si no es concluyente: selección interactiva
    errores = validar_seleccion(df_raw, pais_idx, valor_idx)
    if errores and interactivo:
        print("\n⚠ La detección automática no fue concluyente:")
        for e in errores:
            print("  -", e)
        try:
            pais_idx, valor_idx = seleccion_interactiva(df_raw, pais_idx, valor_idx)
        except (EOFError, ValueError) as e:
            raise ValueError(f"Asignación manual inválida: {e}")
        errores = validar_seleccion(df_raw, pais_idx, valor_idx)

    if errores:
        raise ValueError("Inconsistencias en la selección de columnas:\n  - " + "\n  - ".join(errores))

    # 4) Correspondencia año -> columna
    col_pais = df_raw.columns[pais_idx]
    cols_valor = [df_raw.columns[j] for j in valor_idx]
    mapa_anio = {c: int(_anios_en(c)[0]) for c in cols_valor}
    print(f"\nColumna de países: [{pais_idx}] '{col_pais}'")
    print("Correspondencia año -> columna:")
    for c, a in sorted(mapa_anio.items(), key=lambda x: x[1]):
        print(f"  {a} -> '{c}'")

    # 5) Transformación a formato largo (melt)
    base = df_raw[[col_pais] + cols_valor].copy()
    df_melted = base.melt(id_vars=[col_pais], value_vars=cols_valor,
                          var_name='year_column', value_name='trade_value_usd')
    df_melted['year'] = df_melted['year_column'].map(mapa_anio).astype(int)
    df_melted = df_melted.rename(columns={col_pais: 'partner_country'})

    # 6) Limpieza: conversión numérica, sin valores y sin fila Mundo
    df_melted['trade_value_usd'] = _serie_numerica(df_melted['trade_value_usd'])
    df_melted['partner_country'] = df_melted['partner_country'].astype(str).str.strip()
    out = df_melted[['year', 'partner_country', 'trade_value_usd']].copy()
    out = out.dropna(subset=['trade_value_usd'])
    out = out[~out['partner_country'].map(_norm).isin(NOMBRES_MUNDO)]
    out = out[~out['partner_country'].map(_norm).isin({'', 'nan', 'none'})].reset_index(drop=True)

    if out.empty:
        raise ValueError("El resultado quedó vacío: revise las columnas seleccionadas.")

    print(f"\nDatos procesados: {len(out)} filas")
    print(f"Países únicos: {out['partner_country'].nunique()}")
    print(f"Años disponibles: {sorted(int(a) for a in out['year'].unique())}")
    print("\nPrimeros datos procesados:")
    print(out.head(10))
    return out


# ------------------------------------------------------------
# Ejecución en Google Colab
# ------------------------------------------------------------
if EN_COLAB:
    try:
        get_ipython().system('pip install -q xlrd')       # lectura de .xls
    except Exception:
        pass
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    df = procesar_trademap(filename,
                           pais_manual=COLUMNA_PAIS_MANUAL,
                           valores_manual=COLUMNAS_VALOR_MANUAL)
```

## 4.3. Calcular Market Share, Ranking y aplicar filtros

```
# Calcular market share por año y país
market_evolution = df.groupby(['year', 'partner_country'])['trade_value_usd'].sum().reset_index()

# Calcular market share total por año
yearly_totals = market_evolution.groupby('year')['trade_value_usd'].sum()
market_evolution['market_share'] = market_evolution.apply(
    lambda x: (x['trade_value_usd'] / yearly_totals[x['year']]) * 100, axis=1
)

# Identificar top países
top_countries = df.groupby('partner_country')['trade_value_usd'].sum().nlargest(8).index

# Filtrar solo top países
market_evolution_top = market_evolution[market_evolution['partner_country'].isin(top_countries)]
```

## 4.4.  Generar Tablas Pivot para análisis

```
# TABLA 1: Market Share por año y país
print("📊 TABLA 1: MARKET SHARE POR AÑO (%)")
print("="*60)
pivot_share = market_evolution_top.pivot(index='year', columns='partner_country', values='market_share')
pivot_share_rounded = pivot_share.round(1)
print(pivot_share_rounded.to_string())

# TABLA 2: Valores de comercio (Millones USD)
print("\n\n💰 TABLA 2: VALORES DE COMERCIO (Millones USD)")
print("="*60)
pivot_value = market_evolution_top.pivot(index='year', columns='partner_country', values='trade_value_usd')
pivot_value_millions = (pivot_value / 1000).round(1)  # Convertir a millones
print(pivot_value_millions.to_string())

# TABLA 3: Crecimiento anual por país
print("\n\n📈 TABLA 3: CRECIMIENTO ANUAL COMPUESTO (CAGR %)")
print("="*60)
growth_data = market_evolution_top.groupby('partner_country').apply(
    lambda x: ((x['trade_value_usd'].iloc[-1] / x['trade_value_usd'].iloc[0]) ** (1/(len(x)-1)) - 1) * 100
).reset_index(name='cagr')
growth_summary = growth_data.set_index('partner_country')['cagr'].round(1).sort_values(ascending=False)
print(growth_summary.to_string())

# TABLA 4: Ranking por año
print("\n\n🏆 TABLA 4: RANKING DE PAÍSES POR AÑO (Top 5)")
print("="*60)
for year in sorted(df['year'].unique()):
    year_data = market_evolution_top[market_evolution_top['year'] == year].nlargest(5, 'market_share')
    print(f"\n{year}:")
    for i, (_, row) in enumerate(year_data.iterrows(), 1):
        print(f"  {i}. {row['partner_country']}: {row['market_share']:.1f}% ({row['trade_value_usd']/1000:.0f}M USD)")
```

## 4.5. Generar Gráficos analíticos

```
# Requiere las variables creadas en 4.1-4.4:
#   df, market_evolution_top, pivot_share, pivot_value, growth_data
# No modifica ningún cálculo: solo controla los colores de los gráficos.
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1) PALETAS PREDETERMINADAS (único lugar donde se definen colores)
# ------------------------------------------------------------
PALETAS_PREDETERMINADAS = {
    'Ejecutivo Clásico': {
        'descripcion': 'Azules sobrios, grises y acentos dorados. Informes ejecutivos y corporativos.',
        'colores': ['#1B365D', '#4A7BA7', '#8FB3D9', '#4A5568', '#A0AEC0', '#B8860B', '#D98E32', '#2C5282'],
    },
    'Comercio Global': {
        'descripcion': 'Azules y verdes con acentos vivos. Negocios internacionales y competitividad.',
        'colores': ['#0A3D62', '#1E88C8', '#00A896', '#2E9E5B', '#8BC34A', '#F4A300', '#E4572E', '#5C7AEA'],
    },
    'Corporativo Contemporáneo': {
        'descripcion': 'Neutros, tierras y acentos cálidos. Presentaciones analíticas modernas.',
        'colores': ['#3B3A36', '#8A7968', '#A9552B', '#D1A36B', '#5F6B4A', '#A7B58A', '#D9694F', '#E8C77A'],
    },
}
OPCION_PERSONALIZADA = 'Personalizada (pegar código CSS)'

# Una letra mayor que este umbral de luminosidad se considera demasiado clara sobre fondo blanco
UMBRAL_LUZ = 0.85
MIN_COLORES = 2
_REGEX_HEX = re.compile(r'#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3})(?![0-9a-fA-F])')


# ------------------------------------------------------------
# 2) EXTRACCIÓN Y VALIDACIÓN DE COLORES (el CSS NUNCA se ejecuta)
# ------------------------------------------------------------
def _luz(hex_color):
    r, g, b = mcolors.to_rgb(hex_color)
    return 0.299 * r + 0.587 * g + 0.114 * b


def extraer_colores_css(texto_css):
    """Extrae solo códigos HEX (#RGB, #RRGGBB, #RRGGBBAA) de un texto CSS.
    Devuelve (colores_validos, mensajes). colores_validos == [] si no es utilizable."""
    mensajes = []
    if not texto_css or not texto_css.strip():
        return [], ["El campo está vacío. Pegue el código CSS de su paleta."]

    texto = re.sub(r'/\*.*?\*/', ' ', texto_css, flags=re.S)       # ignorar comentarios CSS
    encontrados = _REGEX_HEX.findall(texto)
    if not encontrados:
        return [], ["No se encontraron colores HEX (ej. #1B365D) en el texto suministrado."]

    colores = []
    for c in encontrados:
        c = c.lower()
        if len(c) == 9:                                              # #rrggbbaa (Coolors): quitar alfa
            c = c[:7]
        elif len(c) == 4:                                            # #rgb -> #rrggbb
            c = '#' + ''.join(ch * 2 for ch in c[1:])
        if c not in colores:                                         # sin duplicados, conserva el orden
            colores.append(c)

    claros = [c for c in colores if _luz(c) > UMBRAL_LUZ]
    utiles = [c for c in colores if c not in claros]
    if claros:
        mensajes.append(f"Se omitieron {len(claros)} color(es) demasiado claros para fondo blanco: {', '.join(claros)}")
    if len(utiles) < MIN_COLORES:
        mensajes.append(f"Se necesitan al menos {MIN_COLORES} colores utilizables y solo hay {len(utiles)}.")
        return [], mensajes
    return utiles, mensajes


# ------------------------------------------------------------
# 3) ESQUEMA CROMÁTICO CENTRALIZADO
# ------------------------------------------------------------
def _mezclar(c1, c2, t):
    a, b = np.array(mcolors.to_rgb(c1)), np.array(mcolors.to_rgb(c2))
    return mcolors.to_hex(a * (1 - t) + b * t)


def _expandir(colores, n):
    """Devuelve n colores; si hay menos, genera tonos intermedios de la misma paleta."""
    if n <= len(colores):
        return list(colores[:n])
    cmap = mcolors.LinearSegmentedColormap.from_list('paleta', colores)
    return [mcolors.to_hex(cmap(x)) for x in np.linspace(0, 1, n)]


def construir_esquema(colores, paises):
    """Asigna un color fijo a cada país y deriva los tonos de texto, ejes y cuadrícula."""
    paises = list(dict.fromkeys(paises))                             # únicos, en orden
    asignados = _expandir(colores, len(paises))
    oscuro = min(colores, key=_luz)
    return {
        'por_pais': dict(zip(paises, asignados)),
        'texto': _mezclar(oscuro, '#000000', 0.25),
        'eje': _mezclar(oscuro, '#ffffff', 0.40),
        'grilla': _mezclar(oscuro, '#ffffff', 0.85),
        'referencia': _mezclar(oscuro, '#ffffff', 0.30),
        'principal': colores[0],
    }


# ------------------------------------------------------------
# 4) GRÁFICOS (mismos cálculos y estructura; solo cambian los colores)
# ------------------------------------------------------------
def dibujar_graficos(esquema):
    c = esquema['por_pais']
    estilo = {
        'text.color': esquema['texto'], 'axes.labelcolor': esquema['texto'],
        'axes.titlecolor': esquema['texto'], 'axes.edgecolor': esquema['eje'],
        'xtick.color': esquema['texto'], 'ytick.color': esquema['texto'],
        'grid.color': esquema['grilla'],
    }
    with plt.rc_context(estilo):
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Market Share Trends con valores en puntos clave
        pivot_share.plot(kind='line', marker='o', ax=axes[0, 0], linewidth=2, markersize=6,
                         color=[c[p] for p in pivot_share.columns])
        axes[0, 0].set_title('Market Share Evolution (%)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Market Share (%)')
        axes[0, 0].grid(True, alpha=0.6)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Añadir valores para Brasil y Colombia (principales)
        for country in ['Brasil', 'Colombia']:
            if country in pivot_share.columns:
                for year in pivot_share.index:
                    value = pivot_share.loc[year, country]
                    if not pd.isna(value):
                        axes[0, 0].annotate(f'{value:.1f}%', (year, value),
                                            textcoords="offset points", xytext=(0, 10),
                                            ha='center', fontsize=8, color=esquema['texto'])

        # Trade Value Evolution
        (pivot_value / 1000).plot(kind='line', marker='s', ax=axes[0, 1], linewidth=2, markersize=6,
                                  color=[c[p] for p in pivot_value.columns])
        axes[0, 1].set_title('Trade Value Evolution (Million USD)', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Million USD')
        axes[0, 1].grid(True, alpha=0.6)

        # Market Share actual con valores
        latest_year = df['year'].max()
        latest_data = market_evolution_top[market_evolution_top['year'] == latest_year].nlargest(8, 'market_share')
        serie_ms = latest_data.set_index('partner_country')['market_share']
        bars = serie_ms.plot(kind='bar', ax=axes[1, 0], color=[c[p] for p in serie_ms.index])
        axes[1, 0].set_title(f'Market Share {latest_year} (%)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Market Share (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Añadir valores encima de las barras
        for i, (country, value) in enumerate(serie_ms.items()):
            axes[1, 0].text(i, value + 0.5, f'{value:.1f}%', ha='center', va='bottom',
                            fontsize=9, color=esquema['texto'])

        # CAGR con valores
        serie_cagr = growth_data.set_index('partner_country')['cagr']
        bars2 = serie_cagr.plot(kind='bar', ax=axes[1, 1], color=[c[p] for p in serie_cagr.index])
        axes[1, 1].set_title('CAGR % (Compound Annual Growth Rate)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('CAGR (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].axhline(y=0, color=esquema['referencia'], linestyle='-', alpha=0.6)

        # Añadir valores encima de las barras
        for i, (country, value) in enumerate(serie_cagr.items()):
            axes[1, 1].text(i, value + 0.5 if value > 0 else value - 1, f'{value:.1f}%',
                            ha='center', va='bottom' if value > 0 else 'top',
                            fontsize=9, color=esquema['texto'])

        plt.tight_layout()
        plt.show()
    return fig, axes, bars, bars2, latest_year, latest_data


def _todos_los_paises():
    """Orden estable de países para que cada uno tenga el mismo color en los 4 gráficos."""
    orden = list(top_countries) if 'top_countries' in globals() else []
    for nombre in (list(pivot_share.columns) + list(pivot_value.columns)
                   + list(growth_data['partner_country'])):
        orden.append(nombre)
    return list(dict.fromkeys(orden))


def generar_con_paleta(colores):
    global fig, axes, bars, bars2, latest_year, latest_data
    esquema = construir_esquema(colores, _todos_los_paises())
    fig, axes, bars, bars2, latest_year, latest_data = dibujar_graficos(esquema)


# ------------------------------------------------------------
# 5) INTERFAZ INTERACTIVA (Google Colab / Jupyter)
# ------------------------------------------------------------
def _html_vista_previa(colores):
    cuadros = ''.join(
        f'<span style="display:inline-block;width:34px;height:20px;background:{c};'
        f'border:1px solid #888;margin-right:2px;" title="{c}"></span>' for c in colores)
    return f'<div style="margin:2px 0 8px 0">{cuadros}</div>'


def mostrar_selector_paletas():
    try:
        import ipywidgets as widgets
        from IPython.display import HTML, clear_output, display
    except ImportError:
        print("ℹ️ No es posible mostrar la interfaz interactiva en este entorno. "
              "Se usa la paleta 'Ejecutivo Clásico'.")
        generar_con_paleta(PALETAS_PREDETERMINADAS['Ejecutivo Clásico']['colores'])
        return

    opciones = list(PALETAS_PREDETERMINADAS) + [OPCION_PERSONALIZADA]
    selector = widgets.RadioButtons(options=opciones, value=opciones[0], description='Paleta:',
                                    layout=widgets.Layout(width='500px'),
                                    style={'description_width': '60px'})
    vista = widgets.HTML()
    css = widgets.Textarea(
        placeholder=("Pegue aquí el código CSS de su paleta (ej. Coolors -> Export -> CSS):\n"
                     ":root {\n  --color-1: #264653;\n  --color-2: #2a9d8f;\n  --color-3: #e76f51;\n}"),
        layout=widgets.Layout(width='620px', height='140px', display='none'))
    boton = widgets.Button(description='Generar gráficos', button_style='success', icon='bar-chart')
    salida = widgets.Output()

    def refrescar(*_):
        nombre = selector.value
        if nombre == OPCION_PERSONALIZADA:
            css.layout.display = 'block'
            vista.value = "<i>Pegue el CSS con colores HEX y pulse «Generar gráficos».</i>"
        else:
            css.layout.display = 'none'
            p = PALETAS_PREDETERMINADAS[nombre]
            vista.value = f"<b>{nombre}</b>: {p['descripcion']}" + _html_vista_previa(p['colores'])

    selector.observe(refrescar, 'value')
    refrescar()

    def al_pulsar(_):
        with salida:
            clear_output(wait=True)
            nombre = selector.value
            if nombre == OPCION_PERSONALIZADA:
                colores, avisos = extraer_colores_css(css.value)
                for a in avisos:
                    print("⚠️", a)
                if not colores:
                    print("❌ No se pudo construir una paleta utilizable. Revise el código CSS e intente de nuevo.")
                    return
                display(HTML("<b>Paleta personalizada detectada:</b>" + _html_vista_previa(colores)))
            else:
                colores = PALETAS_PREDETERMINADAS[nombre]['colores']
            try:
                generar_con_paleta(colores)
            except NameError as e:
                print(f"❌ Faltan variables de los apartados 4.3 y 4.4 ({e}). Ejecútelos primero.")

    boton.on_click(al_pulsar)
    display(widgets.VBox([widgets.HTML("<b>🎨 Seleccione la paleta de colores de los gráficos</b>"),
                          selector, vista, css, boton, salida]))


mostrar_selector_paletas()
# Tras pulsar "Generar gráficos" quedan disponibles: fig, axes, bars, bars2, latest_year, latest_data
```

## 4.7. Generación de Promts para análisis posteriores

**Nota:** *Reemplazar "market_name" (Line 7) por el mercado objetivo (país-destino) antes de proceder con la ejecución del script.*

```
# GENERAR PROMPT AUTOMÁTICO PARA ANÁLISIS
def generate_analysis_prompt(df, market_evolution_top, product_name):
    latest_year = df['year'].max()
    first_year = df['year'].min()

    # Detectar mercado objetivo (desde columnas o contexto)
    market_name = "Estados Unidos"  # Por defecto, se podría dinamizar

    # Generar estadísticas clave
    top3 = market_evolution_top[market_evolution_top['year'] == latest_year].nlargest(3, 'market_share')
    total_market = df[df['year'] == latest_year]['trade_value_usd'].sum()

    prompt = f"""
    Actúa como consultor en inteligencia de mercados y analiza estos datos de market share de {product_name.lower()} en {market_name} ({first_year}-{latest_year}). Proporciona insights estratégicos específicos:

    CONTEXTO DEL MERCADO:
    - Producto: {product_name}
    - Mercado objetivo: {market_name}
    - Período: {first_year}-{latest_year}
    - Tamaño total del mercado ({latest_year}): ${total_market/1000:.0f} millones USD

    TOP 3 LÍDERES ACTUALES ({latest_year}):
    """

    for i, (_, row) in enumerate(top3.iterrows(), 1):
        prompt += f"\n    {i}. {row['partner_country']}: {row['market_share']:.1f}% market share"

    prompt += f"""

    DATOS PARA ANÁLISIS:
    [Pegar aquí las 4 tablas generadas por el script anterior]

    PREGUNTAS ESPECÍFICAS A RESPONDER:

    1. ANÁLISIS COMPETITIVO:
       - ¿Cuáles son las tendencias de los líderes del mercado?
       - ¿Qué países están ganando/perdiendo participación?
       - ¿Hay oportunidades por concentración/fragmentación del mercado?

    2. OPORTUNIDADES DE ENTRADA:
       - ¿Qué nichos o segmentos presentan menos competencia?
       - ¿Existen países que han reducido su participación creando vacíos?
       - ¿Cuál sería el mejor timing para entrar al mercado?

    3. BENCHMARKING REGIONAL:
       - ¿Cómo se comportan países con perfil similar al de tu empresa?
       - ¿Qué países de América Latina tienen mejor performance?
       - ¿Cuáles son las mejores prácticas observables?

    4. ESTRATEGIA DE POSICIONAMIENTO:
       - ¿Competir directamente con líderes o buscar nichos?
       - ¿Qué ventajas competitivas serían más relevantes?
       - ¿Cuál sería un market share objetivo realista en 2-3 años?

    5. RIESGOS Y AMENAZAS:
       - ¿Qué competidores emergentes representan amenazas?
       - ¿Hay signos de saturación o madurez del mercado?
       - ¿Qué factores externos podrían afectar la industria?

    FORMATO DE RESPUESTA:
    - Insights concisos y accionables
    - Recomendaciones estratégicas específicas
    - Identificación de 2-3 oportunidades prioritarias
    - Métricas clave a monitorear
    """

    return prompt

# Generar y mostrar el prompt
analysis_prompt = generate_analysis_prompt(df, market_evolution_top, product_name)
print("="*80)
print("📋 PROMPT PARA ANÁLISIS ESTRATÉGICO:")
print("="*80)
print(analysis_prompt)
print("\n" + "="*80)
print("💡 INSTRUCCIONES:")
print("1. Copia este prompt")
print("2. Pega las 4 tablas de datos generadas arriba")
print("3. Úsalo con cualquier chatbot para obtener insights estratégicos")
print("="*80)
```

## 4.8. Helpers (opcionales) para depuración y personalización
### 4.8.1. Promt para análisis de oportunidad: Origen (Proveedor)-Destino

```
Con base en lo anterior, redacta un texto argumentativo, en tono formal y académico en tercera persona, máximo una cuartilla (250 palabras) donde evidencies los resultados de este estudio de mercados. Enfatiza en el nivel de oportunidad que tendría [COLOMBIA] para ingresar a este mercado desde el punto de vista comercial basándose únicamente en los datos suministrados.
```

### 4.8.2. Promt para personalizar colores en los gráficos

```
Con base en la siguiente paleta de colores:

[CSS.CODE]

Ajusta el siguiente trozo de código en consecuencia, da por hecho que las librerías ya se encuentran instaladas:

[PY.CODE.4.5]
```

### 4.8.3. Ajuste para datos faltantes o sin datos históricos (No calcula CAGR)

```
# TABLA 1: Market Share por año y país
print("📊 TABLA 1: MARKET SHARE POR AÑO (%)")
print("="*60)
pivot_share = market_evolution_top.pivot(index='year', columns='partner_country', values='market_share')
pivot_share_rounded = pivot_share.round(1)
print(pivot_share_rounded.to_string())

# TABLA 2: Valores de comercio (Millones USD)
print("\n\n💰 TABLA 2: VALORES DE COMERCIO (Millones USD)")
print("="*60)
pivot_value = market_evolution_top.pivot(index='year', columns='partner_country', values='trade_value_usd')
pivot_value_millions = (pivot_value / 1000).round(1)  # Convertir a millones
print(pivot_value_millions.to_string())

# TABLA 3: Crecimiento anual por país
# El cálculo del CAGR requiere al menos dos años de datos. Como solo se tiene un año (2024),
# esta tabla no se puede generar con los datos actuales.
# print("\n\n📈 TABLA 3: CRECIMIENTO ANUAL COMPUESTO (CAGR %)")
# print("="*60)
# growth_data = market_evolution_top.groupby('partner_country').apply(
#     lambda x: ((x['trade_value_usd']. iloc[-1] / x['trade_value_usd'].iloc[0]) ** (1/(len(x)-1)) - 1) * 100
# ).reset_index(name='cagr')
# growth_summary = growth_data.set_index('partner_country')['cagr'].round(1).sort_values(ascending=False)
# print(growth_summary.to_string())
print("\n\n📈 TABLA 3: CRECIMIENTO ANUAL COMPUESTO (CAGR %)")
print("="*60)
print("No se puede calcular el CAGR. Se requiere al menos dos años de datos para este cálculo.")

# TABLA 4: Ranking por año
print("\n\n🏆 TABLA 4: RANKING DE PAÍSES POR AÑO (Top 5)")
print("="*60)
for year in sorted(df['year'].unique()):
    year_data = market_evolution_top[market_evolution_top['year'] == year].nlargest(5, 'market_share')
    print(f"\n{year}:")
    for i, (_, row) in enumerate(year_data.iterrows(), 1):
        print(f"  {i}. {row['partner_country']}: {row['market_share']:.1f}% ({row['trade_value_usd']/1000:.0f}M USD)")
```

### 4.8.4. Ajuste para datos faltantes (Calcula CAGR omitiendo datos vacíos o nulos)

```
# TABLA 1: Market Share por año y país
print("📊 TABLA 1: MARKET SHARE POR AÑO (%)")
print("="*60)
pivot_share = market_evolution_top.pivot(index='year', columns='partner_country', values='market_share')
pivot_share_rounded = pivot_share.round(1)
print(pivot_share_rounded.to_string())

# TABLA 2: Valores de comercio (Millones USD)
print("\n\n💰 TABLA 2: VALORES DE COMERCIO (Millones USD)")
print("="*60)
pivot_value = market_evolution_top.pivot(index='year', columns='partner_country', values='trade_value_usd')
pivot_value_millions = (pivot_value / 1000).round(1)  # Convertir a millones
print(pivot_value_millions.to_string())

# TABLA 3: Crecimiento anual por país
print("\n\n📈 TABLA 3: CRECIMIENTO ANUAL COMPUESTO (CAGR %)")
print("="*60)
growth_data = market_evolution_top.groupby('partner_country').apply(
    lambda x: ((x['trade_value_usd'].iloc[-1] / x['trade_value_usd'].iloc[0]) ** (1/(len(x)-1)) - 1) * 100 if len(x) > 1 else np.nan
).reset_index(name='cagr')
growth_summary = growth_data.set_index('partner_country')['cagr'].round(1).sort_values(ascending=False)
print(growth_summary.to_string())

# TABLA 4: Ranking por año
print("\n\n🏆 TABLA 4: RANKING DE PAÍSES POR AÑO (Top 5)")
print("="*60)
for year in sorted(df['year'].unique()):
    year_data = market_evolution_top[market_evolution_top['year'] == year].nlargest(5, 'market_share')
    print(f"\n{year}:")
    for i, (_, row) in enumerate(year_data.iterrows(), 1):
        print(f"  {i}. {row['partner_country']}: {row['market_share']:.1f}% ({row['trade_value_usd']/1000:.0f}M USD)")
```
