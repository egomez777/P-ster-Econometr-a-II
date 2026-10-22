# %% PRIMERA PARTE. Preparación y revisión de la base mensual

# 1. Importar las librerías

import pandas as pd
from pathlib import Path

# 2. Localizar el archivo CSV
ruta = Path(__file__).parent / "real-gold-price.csv"

# 3. Leer los datos
datos = pd.read_csv(ruta, parse_dates=["date"])

# 4. Completar el mes faltante mediante interpolación lineal

# Ordenar las observaciones por fecha
datos = datos.sort_values("date")

# Crear el calendario mensual completo
datos = datos.set_index("date").asfreq("MS")

# Estimar el precio de octubre de 2025
datos["real_gold_price"] = datos["real_gold_price"].interpolate(method="linear")

# Volver a dejar la fecha como columna
datos = datos.reset_index()

# Mostrar el dato estimado
print("\nPrecio estimado para octubre de 2025:")
print(datos.loc[datos["date"] == pd.Timestamp("2025-10-01")])


# 5. Revisar las primeras y últimas observaciones
print(datos.head())
print(datos.tail())

# 6. Contar las observaciones
print("Número de observaciones:", len(datos))

# 7. Comprobar si faltan datos
print(datos.isna().sum())

# 8. Comprobar si falta algún mes en la serie

calendario_mensual = pd.date_range(
    start=datos["date"].min(), end=datos["date"].max(), freq="MS"
)

fechas_faltantes = calendario_mensual.difference(datos["date"])

print("Meses esperados:", len(calendario_mensual))
print("Meses faltantes:", len(fechas_faltantes))
print("Fechas faltantes:", fechas_faltantes)

# %% SEGUNDA PARTE. Construcción de la base trimestral

# 1. Calcular el precio promedio de cada trimestre
datos_trimestrales = (
    datos.set_index("date")["real_gold_price"]
    .resample("QE")
    .mean()
    .rename("real_gold_price")
    .reset_index()
)

# 2. Revisar las primeras observaciones trimestrales
print("\nPrimeros trimestres:")
print(datos_trimestrales.head())

# 3. Revisar las últimas observaciones trimestrales
print("\nÚltimos trimestres:")
print(datos_trimestrales.tail())

# 4. Contar las observaciones trimestrales
print("\nNúmero de observaciones trimestrales:", len(datos_trimestrales))

# 5. Comprobar valores faltantes
print("\nValores faltantes:")
print(datos_trimestrales.isna().sum())

# %% TERCERA PARTE. Análisis gráfico de la serie trimestral

import matplotlib.pyplot as plt

# Graficar la evolución del precio real del oro
plt.figure(figsize=(12, 5))

plt.plot(
    datos_trimestrales["date"], datos_trimestrales["real_gold_price"], linewidth=1.2
)

plt.title("Evolución trimestral del precio real del oro")
plt.xlabel("Fecha")
plt.ylabel("Precio real del oro")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

# %% CUARTA PARTE. Identificación: prueba de estacionariedad ADF

from statsmodels.tsa.stattools import adfuller

# 1. Seleccionar la serie trimestral del oro
serie_oro = datos_trimestrales["real_gold_price"].dropna()

# 2. Aplicar la prueba ADF
resultado_adf = adfuller(serie_oro, regression="c", autolag="AIC")

# 3. Mostrar los resultados
print("Estadístico ADF:", resultado_adf[0])
print("Valor p:", resultado_adf[1])
print("Rezagos utilizados:", resultado_adf[2])
print("Observaciones utilizadas:", resultado_adf[3])

print("\nValores críticos:")
for nivel, valor in resultado_adf[4].items():
    print(f"{nivel}: {valor}")


# Comparar con una especificación que incluye tendencia

resultado_adf_tendencia = adfuller(serie_oro, regression="ct", autolag="AIC")

print("\nADF con constante y tendencia")
print("Estadístico ADF:", resultado_adf_tendencia[0])
print("Valor p:", resultado_adf_tendencia[1])
print("Valores críticos:", resultado_adf_tendencia[4])

# %% QUINTA PARTE. Primera diferencia y prueba de estacionariedad

# 1. Calcular el cambio trimestral del precio real del oro
serie_oro_dif = serie_oro.diff().dropna()

# 2. Aplicar la prueba ADF con constante
resultado_adf_dif = adfuller(serie_oro_dif, regression="c", autolag="AIC")

# 3. Mostrar los resultados
print("ADF de la primera diferencia")
print("Estadístico ADF:", resultado_adf_dif[0])
print("Valor p:", resultado_adf_dif[1])
print("Rezagos utilizados:", resultado_adf_dif[2])
print("Observaciones utilizadas:", resultado_adf_dif[3])

print("\nValores críticos:")
for nivel, valor in resultado_adf_dif[4].items():
    print(f"{nivel}: {valor}")

# %% # %% SEXTA PARTE. Gráfica de la primera diferencia

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.plot(datos_trimestrales["date"].iloc[1:], serie_oro_dif, linewidth=1.2)

plt.axhline(y=0, linestyle="--", linewidth=1)

plt.title("Primera diferencia del precio real del oro")
plt.xlabel("Fecha")
plt.ylabel("Cambio trimestral del precio")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

# %% SEPTIMA PARTE. Identificación: ACF y PACF de la primera diferencia

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Crear una figura con dos gráficas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Función de autocorrelación
plot_acf(serie_oro_dif, lags=20, zero=False, ax=axes[0])
axes[0].set_title("ACF de la primera diferencia")

# Función de autocorrelación parcial
plot_pacf(serie_oro_dif, lags=20, method="ywm", zero=False, ax=axes[1])
axes[1].set_title("PACF de la primera diferencia")

plt.tight_layout()
plt.show()

# %%
