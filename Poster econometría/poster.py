# %% PRIMERA PARTE. Preparación y revisión de la base mensual

# 1. Importar las librerías
import pandas as pd
from pathlib import Path

# 2. Localizar y cargar la base
ruta = Path(__file__).parent / "gold_inflation_adjusted.csv"

datos = pd.read_csv(ruta)

# 3. Convertir las fechas y organizar las columnas
datos["Date"] = pd.to_datetime(datos["Date"], format="%m/%d/%Y")

datos = datos.rename(columns={"Date": "date", "Value": "real_gold_price"})

datos = datos.sort_values("date")

# 4. Seleccionar los últimos 50 años completos
datos = datos.loc[
    (datos["date"] >= "1976-10-01") & (datos["date"] <= "2026-09-01")
].copy()

# 5. Revisar las observaciones
print("Primeras observaciones:")
print(datos.head())

print("\nÚltimas observaciones:")
print(datos.tail())

# 6. Verificar el número de observaciones
print("\nNúmero de observaciones:", len(datos))

# 7. Comprobar valores faltantes
print("\nValores faltantes:")
print(datos.isna().sum())

# 8. Comprobar si falta algún mes
calendario = pd.date_range(
    start=datos["date"].min(), end=datos["date"].max(), freq="MS"
)

fechas_faltantes = calendario.difference(datos["date"])

print("\nMeses esperados:", len(calendario))
print("Meses faltantes:", len(fechas_faltantes))
print("Fechas faltantes:", fechas_faltantes)

# %% SEGUNDA PARTE. Construcción y gráfica de la serie trimestral

import matplotlib.pyplot as plt

# 1. Calcular el promedio de cada trimestre
datos_trimestrales = (
    datos.set_index("date")["real_gold_price"]
    .resample("QE")
    .mean()
    .rename("real_gold_price")
    .reset_index()
)

# 2. Revisar los resultados
print("Primeros trimestres:")
print(datos_trimestrales.head())

print("\nÚltimos trimestres:")
print(datos_trimestrales.tail())

print("\nNúmero de observaciones:", len(datos_trimestrales))

print("\nValores faltantes:")
print(datos_trimestrales.isna().sum())

# 3. Graficar la serie trimestral
plt.figure(figsize=(12, 5))

plt.plot(datos_trimestrales["date"], datos_trimestrales["real_gold_price"], linewidth=1)

plt.title("Precio real trimestral del oro (1976-2026)")
plt.xlabel("Fecha")
plt.ylabel("Precio en dólares ajustados por inflación")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% TERCERA PARTE. Identificación: prueba ADF trimestral

from statsmodels.tsa.stattools import adfuller

# 1. Seleccionar el precio real trimestral del oro
serie_oro = datos_trimestrales["real_gold_price"].dropna()

# 2. ADF con constante
resultado_adf = adfuller(serie_oro, regression="c", autolag="AIC")

print("ADF trimestral con constante")
print("Estadístico ADF:", resultado_adf[0])
print("Valor p:", resultado_adf[1])
print("Rezagos utilizados:", resultado_adf[2])
print("Valores críticos:", resultado_adf[4])

# 3. ADF con constante y tendencia
resultado_adf_tendencia = adfuller(serie_oro, regression="ct", autolag="AIC")

print("\nADF trimestral con constante y tendencia")
print("Estadístico ADF:", resultado_adf_tendencia[0])
print("Valor p:", resultado_adf_tendencia[1])
print("Rezagos utilizados:", resultado_adf_tendencia[2])
print("Valores críticos:", resultado_adf_tendencia[4])

# %% CUARTA PARTE. Primera diferencia y prueba ADF trimestral

from statsmodels.tsa.stattools import adfuller

# 1. Calcular el cambio trimestral del precio real del oro
serie_oro_dif = serie_oro.diff().dropna()

# 2. Aplicar la prueba ADF con constante
resultado_adf_dif = adfuller(serie_oro_dif, regression="c", autolag="AIC")

# 3. Mostrar los resultados
print("ADF de la primera diferencia trimestral")
print("Estadístico ADF:", resultado_adf_dif[0])
print("Valor p:", resultado_adf_dif[1])
print("Rezagos utilizados:", resultado_adf_dif[2])
print("Observaciones utilizadas:", resultado_adf_dif[3])

print("\nValores críticos:")
for nivel, valor in resultado_adf_dif[4].items():
    print(f"{nivel}: {valor}")

# %% QUINTA PARTE. Gráfica de la primera diferencia trimestral

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.plot(datos_trimestrales["date"].iloc[1:], serie_oro_dif, linewidth=1)

plt.axhline(y=0, linestyle="--", linewidth=1)

plt.title("Primera diferencia trimestral del precio real del oro")
plt.xlabel("Fecha")
plt.ylabel("Cambio trimestral del precio")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
# %% SEXTA PARTE. Identificación: ACF y PACF de la primera diferencia trimestral

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Crear las dos gráficas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ACF: autocorrelación
plot_acf(serie_oro_dif, lags=20, zero=False, ax=axes[0])
axes[0].set_title("ACF de la primera diferencia trimestral")

# PACF: autocorrelación parcial
plot_pacf(serie_oro_dif, lags=20, method="ywm", zero=False, ax=axes[1])
axes[1].set_title("PACF de la primera diferencia trimestral")

plt.tight_layout()
plt.show()

# %% SEPTIMA PARTE. serie trimestral con su índice de fechas
serie_oro = datos_trimestrales.set_index("date")["real_gold_price"].asfreq("QE")

from statsmodels.tsa.arima.model import ARIMA

# 2. Estimar ARIMA(1,1,0)
modelo_110 = ARIMA(serie_oro, order=(1, 1, 0), trend="t").fit()

# 3. Estimar ARIMA(0,1,1)
modelo_011 = ARIMA(serie_oro, order=(0, 1, 1), trend="t").fit()

# 4. Estimar ARIMA(1,1,1)
modelo_111 = ARIMA(serie_oro, order=(1, 1, 1), trend="t").fit()

# 5. Comparar los criterios de información
print("Comparación de modelos ARIMA")
print("-----------------------------------")

print("ARIMA(1,1,0) - AIC:", modelo_110.aic, "| BIC:", modelo_110.bic)

print("ARIMA(0,1,1) - AIC:", modelo_011.aic, "| BIC:", modelo_011.bic)

print("ARIMA(1,1,1) - AIC:", modelo_111.aic, "| BIC:", modelo_111.bic)
# %% OCTAVA PARTE. Validación del modelo ARIMA(1,1,0)

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# 1. Obtener los residuos del modelo
residuos = modelo_110.resid

# 2. Graficar los residuos y su autocorrelación
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(residuos, linewidth=1)
axes[0].axhline(y=0, linestyle="--", linewidth=1)
axes[0].set_title("Residuos del ARIMA(1,1,0)")
axes[0].set_xlabel("Observación")
axes[0].set_ylabel("Residuo")

plot_acf(residuos, lags=20, zero=False, ax=axes[1])
axes[1].set_title("ACF de los residuos")

plt.tight_layout()
plt.show()

# 3. Prueba de Ljung-Box
resultado_ljung_box = acorr_ljungbox(residuos, lags=[8, 12], model_df=1, return_df=True)

print("\nPrueba de Ljung-Box:")
print(resultado_ljung_box)

# %% NOVENA PARTE. Validación: distribución de los residuos

import matplotlib.pyplot as plt
from scipy.stats import probplot, jarque_bera

# 1. Histograma de los residuos
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(residuos, bins=20, edgecolor="black")
axes[0].set_title("Distribución de los residuos")
axes[0].set_xlabel("Residuo")
axes[0].set_ylabel("Frecuencia")

# 2. Gráfica Q-Q frente a una distribución normal
probplot(residuos, dist="norm", plot=axes[1])
axes[1].set_title("Gráfica Q-Q de los residuos")

plt.tight_layout()
plt.show()

# 3. Prueba de normalidad de Jarque-Bera
resultado_jb = jarque_bera(residuos)

print("Prueba de Jarque-Bera")
print("Estadístico:", resultado_jb.statistic)
print("Valor p:", resultado_jb.pvalue)

# %% DECIMA PARTE. Validación: prueba ARCH sobre los residuos

from statsmodels.stats.diagnostic import het_arch

# Aplicar la prueba ARCH LM
resultado_arch = het_arch(residuos, nlags=8)

print("Prueba ARCH LM")
print("Estadístico LM:", resultado_arch[0])
print("Valor p:", resultado_arch[1])

# Interpretación
if resultado_arch[1] < 0.05:
    print("Se rechaza H0: hay evidencia de efectos ARCH.")
else:
    print("No se rechaza H0: no hay evidencia suficiente de efectos ARCH.")

# %% ONCEAVA PARTE. Comparación de los diagnósticos de los modelos

from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera
import pandas as pd

# 1. Organizar los tres modelos candidatos
modelos = {
    "ARIMA(1,1,0)": (modelo_110, 1),
    "ARIMA(0,1,1)": (modelo_011, 1),
    "ARIMA(1,1,1)": (modelo_111, 2),
}

# 2. Aplicar las pruebas a los residuos de cada modelo
resultados = []

for nombre, (modelo, grados_libertad) in modelos.items():
    residuos_modelo = modelo.resid.dropna()

    # Prueba de autocorrelación: Ljung-Box
    lb = acorr_ljungbox(
        residuos_modelo, lags=[8], model_df=grados_libertad, return_df=True
    )

    # Prueba de normalidad: Jarque-Bera
    jb = jarque_bera(residuos_modelo)

    # Prueba de heterocedasticidad: ARCH LM
    arch = het_arch(residuos_modelo, nlags=8)

    resultados.append(
        {
            "Modelo": nombre,
            "AIC": modelo.aic,
            "BIC": modelo.bic,
            "Ljung-Box p": lb["lb_pvalue"].iloc[0],
            "Jarque-Bera p": jb.pvalue,
            "ARCH p": arch[1],
        }
    )

# 3. Mostrar la comparación
tabla_diagnosticos = pd.DataFrame(resultados)

print(tabla_diagnosticos.to_string(index=False))

# %% DOCEAVA PARTE. Transformación logarítmica

import numpy as np
import matplotlib.pyplot as plt

# 1. Recuperar la serie trimestral
serie_oro = (
    datos_trimestrales
    .set_index("date")["real_gold_price"]
    .asfreq("QE")
)

# 2. Aplicar la transformación logarítmica
serie_oro_log = np.log(serie_oro)

# 3. Graficar la serie original y la transformada
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Precio real en niveles
axes[0].plot(serie_oro.index, serie_oro, linewidth=1)
axes[0].set_title("Precio real trimestral del oro en niveles")
axes[0].set_ylabel("Precio real")
axes[0].grid(True, alpha=0.3)

# Precio real en logaritmos
axes[1].plot(serie_oro_log.index, serie_oro_log, linewidth=1)
axes[1].set_title("Logaritmo del precio real trimestral del oro")
axes[1].set_xlabel("Fecha")
axes[1].set_ylabel("Logaritmo del precio")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# 4. Comprobar la estacionariedad del logaritmo

from statsmodels.tsa.stattools import adfuller

# ADF con constante
adf_log_c = adfuller(
    serie_oro_log,
    regression="c",
    autolag="AIC"
)

print("\nADF del logaritmo con constante")
print("Estadístico:", adf_log_c[0])
print("Valor p:", adf_log_c[1])
print("Valores críticos:", adf_log_c[4])


# ADF con constante y tendencia
adf_log_ct = adfuller(
    serie_oro_log,
    regression="ct",
    autolag="AIC"
)

print("\nADF del logaritmo con constante y tendencia")
print("Estadístico:", adf_log_ct[0])
print("Valor p:", adf_log_ct[1])
print("Valores críticos:", adf_log_ct[4])

# 5. Primera diferencia del logaritmo

serie_oro_log_dif = serie_oro_log.diff().dropna()

# 6. Graficar la primera diferencia del logaritmo

plt.figure(figsize=(12, 5))

plt.plot(
    serie_oro_log.index[1:],
    serie_oro_log_dif,
    linewidth=1
)

plt.axhline(y=0, linestyle="--", linewidth=1)

plt.title("Primera diferencia del logaritmo del precio real del oro")
plt.xlabel("Fecha")
plt.ylabel("Cambio del logaritmo")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

# 7. Prueba ADF de la primera diferencia del logaritmo

resultado_log_dif = adfuller(
    serie_oro_log_dif,
    regression="c",
    autolag="AIC"
)

print("ADF de la primera diferencia del logaritmo")
print("Estadístico ADF:", resultado_log_dif[0])
print("Valor p:", resultado_log_dif[1])
print("Rezagos utilizados:", resultado_log_dif[2])
print("Valores críticos:", resultado_log_dif[4])

# %% TRECEAVA PARTE. Identificación del logaritmo diferenciado: ACF y PACF

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Crear las dos gráficas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ACF de la primera diferencia del logaritmo
plot_acf(serie_oro_log_dif, lags=20, zero=False, ax=axes[0])
axes[0].set_title("ACF del logaritmo diferenciado")

# PACF de la primera diferencia del logaritmo
plot_pacf(
    serie_oro_log_dif,
    lags=20,
    method="ywm",
    zero=False,
    ax=axes[1]
)
axes[1].set_title("PACF del logaritmo diferenciado")

plt.tight_layout()
plt.show()

# %% CATORCEAVA PARTE. Estimación y comparación de modelos ARIMA

from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# 1. Definir todos los modelos candidatos
# Definir los modelos candidatos
# trend="n": sin deriva
# trend="t": con deriva

especificaciones = [
    ("ARIMA(1,1,0) sin deriva", (1, 1, 0), "n"),
    ("ARIMA(1,1,0) con deriva", (1, 1, 0), "t"),

    ("ARIMA(0,1,1) sin deriva", (0, 1, 1), "n"),
    ("ARIMA(0,1,1) con deriva", (0, 1, 1), "t"),

    ("ARIMA(1,1,1) sin deriva", (1, 1, 1), "n"),
    ("ARIMA(1,1,1) con deriva", (1, 1, 1), "t")
]

# 2. Estimar todos los modelos sobre la misma serie
modelos_candidatos = {}
resultados_estimacion = []

for nombre, orden, tendencia in especificaciones:

    modelo = ARIMA(
        serie_oro_log,
        order=orden,
        trend=tendencia
    ).fit()

    modelos_candidatos[nombre] = modelo

    resultados_estimacion.append({
        "Modelo": nombre,
        "Log-verosimilitud": modelo.llf,
        "AIC": modelo.aic,
        "BIC": modelo.bic
    })

# 3. Comparar los seis modelos
tabla_modelos = pd.DataFrame(resultados_estimacion)

print("COMPARACIÓN DE LOS SEIS MODELOS CANDIDATOS")
print("==========================================")

print(
    tabla_modelos
    .sort_values("BIC")
    .to_string(index=False)
)

# %% QUINCEAVA PARTE. Diagnóstico de residuos estandarizados

from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera
import numpy as np
import pandas as pd

# 1. Preparar los resultados
resultados_diagnostico = []

# 2. Evaluar los seis modelos
for nombre, orden, tendencia in especificaciones:

    modelo = modelos_candidatos[nombre]

    # Obtener residuos estandarizados de un paso adelante
    residuos = pd.Series(
        modelo.filter_results.standardized_forecasts_error[0]
    )

    # Excluir valores no finitos
    residuos = residuos.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    # Grados de libertad para Ljung-Box
    grados_libertad = orden[0] + orden[2]

    # Prueba de autocorrelación
    lb = acorr_ljungbox(
        residuos,
        lags=[8],
        model_df=grados_libertad,
        return_df=True
    )

    # Prueba de normalidad
    jb = jarque_bera(residuos)

    # Prueba ARCH-LM
    arch = het_arch(residuos, nlags=8)

    # Guardar resultados
    resultados_diagnostico.append({
        "Modelo": nombre,
        "AIC": modelo.aic,
        "BIC": modelo.bic,
        "Ljung-Box p": lb["lb_pvalue"].iloc[0],
        "Jarque-Bera p": jb.pvalue,
        "ARCH p": arch[1]
    })

# 3. Mostrar resultados
tabla_diagnosticos = pd.DataFrame(resultados_diagnostico)

print("DIAGNÓSTICO CON RESIDUOS ESTANDARIZADOS")
print(tabla_diagnosticos.sort_values("BIC").to_string(index=False))
# %% DOCEAVA PARTE. Estimación del Modelo ARIMA (1,1,0)

# 1. Crear un objeto de la clase ARIMA (Asociado a un AR(1))

import numpy as np

modelo_ar1 = ARIMA(np.log(serie_oro), order=(1, 1, 0), trend="t").fit()

# 2. Objeto tipo ARIMA del paquete statmodels

type(modelo_ar1)

print(modelo_ar1.summary())

# %%
# Dado que en la parte ar.L1 0.3300>0.000, es altamente significativo, lo que confirma que el precio del oro en el trimestre anterior tiene un impacto real en el precio actual.
# %%
# Dado que en la tendencia x1 0.0087<0.239, la tendencia lineal no es estadísticamente significativa. Se intentará hacer una versión sin este parámetro para ver si el modelo mejora.
# %%
# Cumplimiento del Ruido Blanco: Dado que la prueba de Ljung-Box con Prob(Q) es 0.69>0.05, indica que no queda información residual oculta; el modelo capturó bien la dinámica temporal y no hay autocorrelación serial.
# %%
# La prueba de Jarque-Bera con Prob(JB) es 0.13>0.05, indica que los residuos se distribuyen de forma normal.
# %%
# La prueba de Heterocedasticidad con Prob(H) es 0.09>0.05, indica que está cerca del límite pero sí lo supera, por lo que se asume que la volatilidad de los errores es estable a lo largo de las observaciones.

# 3. Prueba del modelo sin tendencia.

# 1. Estimar el modelo sin la tendencia
modelo_ar1_sin_tendencia = ARIMA(np.log(serie_oro), order=(1, 1, 0), trend="n").fit()

# 2. Imprimir el nuevo resumen estadístico
print(modelo_ar1_sin_tendencia.summary())

# El modelo sin tendencia es superior al modelo anterior porque al eliminar la variable que no aporta información sustancial, se hace más parsimonioso sin sacrificar poder explicativo.

# El peso del rezago trimestral sube ligeramente, los criterios de información son menores, y se siguen manteniendo los resultados de los supuestos.
