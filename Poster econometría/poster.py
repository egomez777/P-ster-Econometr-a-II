# %% PRIMERA PARTE. Preparación y revisión de la base mensual

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import jarque_bera, probplot
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# 1. Localizar y cargar la base usando una ruta relativa
ruta = Path(__file__).resolve().parent / "gold_inflation_adjusted.csv"

if not ruta.exists():
    raise FileNotFoundError(
        "No se encontró 'gold_inflation_adjusted.csv'. "
        "Guarda el CSV en la misma carpeta que este script."
    )

datos = pd.read_csv(ruta)

# 2. Convertir las fechas y organizar las columnas
datos["Date"] = pd.to_datetime(datos["Date"], format="%m/%d/%Y")
datos = datos.rename(columns={"Date": "date", "Value": "real_gold_price"})
datos = datos.sort_values("date")

# 3. Seleccionar el período de octubre de 1976 a septiembre de 2026
datos = datos.loc[
    (datos["date"] >= "1976-10-01") & (datos["date"] <= "2026-09-01")
].copy()

# 4. Revisar las observaciones mensuales
print("PRIMERAS OBSERVACIONES MENSUALES")
print(datos.head())
print("\nÚLTIMAS OBSERVACIONES MENSUALES")
print(datos.tail())
print("\nNúmero de observaciones mensuales:", len(datos))
print("\nValores faltantes por columna:")
print(datos.isna().sum())

# 5. Comprobar si falta algún mes
calendario = pd.date_range(
    start=datos["date"].min(), end=datos["date"].max(), freq="MS"
)
fechas_faltantes = calendario.difference(datos["date"])
print("\nMeses esperados:", len(calendario))
print("Meses faltantes:", len(fechas_faltantes))
if len(fechas_faltantes) > 0:
    print(fechas_faltantes)

# %% SEGUNDA PARTE. Construcción y exploración de la serie trimestral original

# 1. Calcular el promedio de cada trimestre a partir de los datos mensuales
datos_trimestrales = (
    datos.set_index("date")["real_gold_price"]
    .resample("QE")
    .mean()
    .rename("real_gold_price")
    .to_frame()
)

# 2. Crear la serie trimestral con índice de fechas
serie_oro = datos_trimestrales["real_gold_price"].asfreq("QE")

print("\nPRIMEROS TRIMESTRES")
print(datos_trimestrales.head())
print("\nÚLTIMOS TRIMESTRES")
print(datos_trimestrales.tail())
print("\nNúmero de observaciones trimestrales:", len(serie_oro))
print("Valores faltantes en la serie trimestral:", serie_oro.isna().sum())

# 3. Graficar la serie original en niveles
plt.figure(figsize=(12, 5))
plt.plot(serie_oro.index, serie_oro, linewidth=1)
plt.title("Precio real trimestral del oro en niveles (1976-2026)")
plt.xlabel("Fecha")
plt.ylabel("Precio real del oro")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Graficar la FAC y FACP de la serie original en niveles
# Estas gráficas ayudan a visualizar la persistencia de la serie original.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(serie_oro, lags=20, zero=False, ax=axes[0])
axes[0].set_title("FAC de la serie original")
axes[0].set_xlabel("Rezago trimestral")

plot_pacf(serie_oro, lags=20, method="ywm", zero=False, ax=axes[1])
axes[1].set_title("FACP de la serie original")
axes[1].set_xlabel("Rezago trimestral")

plt.tight_layout()
plt.show()
# %% TERCERA PARTE. Identificación inicial de la no estacionariedad

# 1. ADF de la serie original en niveles con constante
adf_nivel_c = adfuller(serie_oro, regression="c", autolag="AIC")
print("ADF DE LA SERIE ORIGINAL EN NIVELES: CON CONSTANTE")
print("Estadístico ADF:", adf_nivel_c[0])
print("Valor p:", adf_nivel_c[1])
print("Rezagos utilizados:", adf_nivel_c[2])
print("Valores críticos:", adf_nivel_c[4])

# 2. ADF de la serie original en niveles con constante y tendencia
adf_nivel_ct = adfuller(serie_oro, regression="ct", autolag="AIC")
print("\nADF DE LA SERIE ORIGINAL EN NIVELES: CON CONSTANTE Y TENDENCIA")
print("Estadístico ADF:", adf_nivel_ct[0])
print("Valor p:", adf_nivel_ct[1])
print("Rezagos utilizados:", adf_nivel_ct[2])
print("Valores críticos:", adf_nivel_ct[4])

# %% CUARTA PARTE. Primera diferencia de la serie original

# 1. Calcular la primera diferencia del precio real del oro
serie_oro_dif = serie_oro.diff().dropna()

# 2. Graficar la primera diferencia
plt.figure(figsize=(12, 5))
plt.plot(serie_oro_dif.index, serie_oro_dif, linewidth=1)
plt.axhline(y=0, linestyle="--", linewidth=1, color="black")
plt.title("Primera diferencia del precio real trimestral del oro")
plt.xlabel("Fecha")
plt.ylabel("Cambio trimestral del precio")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. ADF de la primera diferencia del precio original
adf_dif = adfuller(serie_oro_dif, regression="c", autolag="AIC")
print("ADF DE LA PRIMERA DIFERENCIA DEL PRECIO ORIGINAL")
print("Estadístico ADF:", adf_dif[0])
print("Valor p:", adf_dif[1])
print("Rezagos utilizados:", adf_dif[2])
print("Observaciones utilizadas:", adf_dif[3])
print("Valores críticos:", adf_dif[4])

# 4. Explorar visualmente si la variabilidad de los cambios parece constante
# La desviación estándar móvil resume la dispersión en ventanas de 20 trimestres.
# Es una herramienta exploratoria, no una prueba formal de heterocedasticidad.
desviacion_movil = serie_oro_dif.rolling(window=20).std()

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(serie_oro_dif.index, serie_oro_dif, linewidth=1)
axes[0].axhline(0, linestyle="--", linewidth=1, color="black")
axes[0].set_title("Cambios trimestrales del precio real del oro")
axes[0].set_ylabel("Primera diferencia")
axes[0].grid(True, alpha=0.3)

axes[1].plot(desviacion_movil.index, desviacion_movil, linewidth=1.2)
axes[1].set_title("Desviación estándar móvil de los cambios (ventana de 20 trimestres)")
axes[1].set_xlabel("Fecha")
axes[1].set_ylabel("Desviación estándar móvil")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% QUINTA PARTE. Transformación logarítmica de la serie original

# La transformación logarítmica se aplica a la serie original positiva,
# no a la primera diferencia, que puede contener valores negativos.
if (serie_oro <= 0).any():
    raise ValueError(
        "La serie debe contener valores positivos para aplicar logaritmos."
    )

serie_oro_log = np.log(serie_oro)

# 1. Comparar visualmente los niveles originales con sus logaritmos
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(serie_oro.index, serie_oro, linewidth=1)
axes[0].set_title("Precio real trimestral del oro en niveles")
axes[0].set_ylabel("Precio real")
axes[0].grid(True, alpha=0.3)

axes[1].plot(serie_oro_log.index, serie_oro_log, linewidth=1)
axes[1].set_title("Logaritmo del precio real trimestral del oro")
axes[1].set_xlabel("Fecha")
axes[1].set_ylabel("Logaritmo del precio")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 2. ADF del logaritmo en niveles con constante
adf_log_c = adfuller(serie_oro_log, regression="c", autolag="AIC")
print("ADF DEL LOGARITMO EN NIVELES: CON CONSTANTE")
print("Estadístico ADF:", adf_log_c[0])
print("Valor p:", adf_log_c[1])
print("Rezagos utilizados:", adf_log_c[2])
print("Valores críticos:", adf_log_c[4])

# 3. ADF del logaritmo en niveles con constante y tendencia
adf_log_ct = adfuller(serie_oro_log, regression="ct", autolag="AIC")
print("\nADF DEL LOGARITMO EN NIVELES: CON CONSTANTE Y TENDENCIA")
print("Estadístico ADF:", adf_log_ct[0])
print("Valor p:", adf_log_ct[1])
print("Rezagos utilizados:", adf_log_ct[2])
print("Valores críticos:", adf_log_ct[4])

# %% SEXTA PARTE. Primera diferencia del logaritmo y estacionariedad

# 1. Diferenciar el logaritmo; aproxima la tasa de crecimiento del precio
serie_oro_log_dif = serie_oro_log.diff().dropna()

# 2. Graficar la primera diferencia del logaritmo
plt.figure(figsize=(12, 5))
plt.plot(serie_oro_log_dif.index, serie_oro_log_dif, linewidth=1)
plt.axhline(y=0, linestyle="--", linewidth=1, color="black")
plt.title("Primera diferencia del logaritmo del precio real del oro")
plt.xlabel("Fecha")
plt.ylabel("Cambio del logaritmo")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. ADF de la primera diferencia del logaritmo
adf_log_dif = adfuller(serie_oro_log_dif, regression="c", autolag="AIC")
print("ADF DE LA PRIMERA DIFERENCIA DEL LOGARITMO")
print("Estadístico ADF:", adf_log_dif[0])
print("Valor p:", adf_log_dif[1])
print("Rezagos utilizados:", adf_log_dif[2])
print("Observaciones utilizadas:", adf_log_dif[3])
print("Valores críticos:", adf_log_dif[4])

# %% SÉPTIMA PARTE. Identificación: ACF y PACF del logaritmo diferenciado

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(serie_oro_log_dif, lags=20, zero=False, ax=axes[0])
axes[0].set_title("ACF de la primera diferencia del logaritmo")
plot_pacf(serie_oro_log_dif, lags=20, method="ywm", zero=False, ax=axes[1])
axes[1].set_title("PACF de la primera diferencia del logaritmo")
plt.tight_layout()
plt.show()

# %% OCTAVA PARTE. Revisión exploratoria de estacionalidad trimestral

# En datos trimestrales, revisar los múltiplos de cuatro rezagos.
valores_acf = acf(serie_oro_log_dif, nlags=20, fft=False)
rezagos_estacionales = [4, 8, 12, 16, 20]
limite_acf = 1.96 / np.sqrt(len(serie_oro_log_dif))

tabla_estacionalidad = pd.DataFrame(
    {
        "Rezago": rezagos_estacionales,
        "ACF": [valores_acf[rezago] for rezago in rezagos_estacionales],
    }
)
tabla_estacionalidad["Fuera de la banda"] = (
    tabla_estacionalidad["ACF"].abs() > limite_acf
)

print("REVISIÓN EXPLORATORIA DE LA ESTACIONALIDAD")
print(tabla_estacionalidad.round(4).to_string(index=False))
print(f"Banda aproximada del 95 %: ±{limite_acf:.4f}")

# %% NOVENA PARTE. Estimación y comparación de modelos ARIMA

# trend="n": sin deriva; trend="t": con deriva.
especificaciones = [
    ("ARIMA(1,1,0) sin deriva", (1, 1, 0), "n"),
    ("ARIMA(1,1,0) con deriva", (1, 1, 0), "t"),
    ("ARIMA(0,1,1) sin deriva", (0, 1, 1), "n"),
    ("ARIMA(0,1,1) con deriva", (0, 1, 1), "t"),
    ("ARIMA(1,1,1) sin deriva", (1, 1, 1), "n"),
    ("ARIMA(1,1,1) con deriva", (1, 1, 1), "t"),
]

modelos_candidatos = {}
resultados_estimacion = []

for nombre, orden, tendencia in especificaciones:
    modelo = ARIMA(serie_oro_log, order=orden, trend=tendencia).fit()
    modelos_candidatos[nombre] = modelo
    resultados_estimacion.append(
        {
            "Modelo": nombre,
            "Log-verosimilitud": modelo.llf,
            "AIC": modelo.aic,
            "BIC": modelo.bic,
        }
    )

tabla_modelos = pd.DataFrame(resultados_estimacion)
print("COMPARACIÓN DE LOS SEIS MODELOS CANDIDATOS")
print(tabla_modelos.sort_values("BIC").round(4).to_string(index=False))

# %% DÉCIMA PARTE. Diagnóstico estadístico de los modelos ARIMA

resultados_diagnosticos = []

for nombre, modelo in modelos_candidatos.items():
    # Residuos originales, sin estandarizar.
    residuos = pd.Series(modelo.resid).dropna()

    # Se omite el periodo inicial señalado por loglikelihood_burn solo del
    # diagnóstico; no se elimina ninguna observación de la serie ni se reestima.
    residuos = residuos.iloc[modelo.loglikelihood_burn :]

    orden = next(
        orden
        for nombre_esp, orden, tendencia in especificaciones
        if nombre_esp == nombre
    )
    p, d, q = orden

    # Ljung-Box: H0 = no hay autocorrelación residual hasta el rezago evaluado.
    lb = acorr_ljungbox(residuos, lags=[8, 12], model_df=p + q, return_df=True)

    # Jarque-Bera: H0 = los residuos siguen una distribución normal.
    jb = jarque_bera(residuos)

    # ARCH LM: H0 = no hay efectos ARCH.
    arch = het_arch(residuos, nlags=8)

    resultados_diagnosticos.append(
        {
            "Modelo": nombre,
            "Ljung-Box p (8)": lb.loc[8, "lb_pvalue"],
            "Ljung-Box p (12)": lb.loc[12, "lb_pvalue"],
            "Jarque-Bera p": jb.pvalue,
            "ARCH p": arch[1],
        }
    )

tabla_diagnosticos = pd.DataFrame(resultados_diagnosticos)
print("DIAGNÓSTICO DE LOS MODELOS ARIMA")
print(
    tabla_diagnosticos.to_string(
        index=False,
        formatters={
            "Ljung-Box p (8)": lambda x: f"{x:.4f}",
            "Ljung-Box p (12)": lambda x: f"{x:.4f}",
            "Jarque-Bera p": lambda x: f"{x:.4g}",
            "ARCH p": lambda x: f"{x:.3e}",
        },
    )
)

# %% UNDÉCIMA PARTE. Diagnóstico gráfico del modelo candidato

# Candidato provisional: ARIMA(1,1,0) sin deriva, elegido inicialmente por BIC.
modelo_candidato = modelos_candidatos["ARIMA(1,1,0) sin deriva"]
residuos_candidato = pd.Series(modelo_candidato.resid).dropna()
residuos_candidato = residuos_candidato.iloc[modelo_candidato.loglikelihood_burn :]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 1. Residuos a lo largo del tiempo
axes[0, 0].plot(residuos_candidato.index, residuos_candidato, linewidth=1)
axes[0, 0].axhline(0, color="black", linestyle="--")
axes[0, 0].set_title("Residuos del ARIMA(1,1,0) sin deriva")
axes[0, 0].set_xlabel("Fecha")
axes[0, 0].set_ylabel("Residuo")

# 2. ACF de los residuos
plot_acf(residuos_candidato, lags=20, zero=False, ax=axes[0, 1])
axes[0, 1].set_title("ACF de los residuos")

# 3. ACF de los residuos al cuadrado: dependencia en la volatilidad
plot_acf(residuos_candidato**2, lags=20, zero=False, ax=axes[1, 0])
axes[1, 0].set_title("ACF de los residuos al cuadrado")

# 4. Gráfica Q-Q para examinar la normalidad
probplot(residuos_candidato, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Gráfica Q-Q de los residuos")

plt.tight_layout()
plt.show()

# %% DUODÉCIMA PARTE. Pronóstico de diez trimestres

# Modelo principal provisional: ARIMA(1,1,0) sin deriva.
modelo_final = modelos_candidatos["ARIMA(1,1,0) sin deriva"]
resultado_pronostico = modelo_final.get_forecast(steps=10)

# Pronóstico e intervalos en la escala del logaritmo.
pronostico_log = resultado_pronostico.predicted_mean
intervalos_log = resultado_pronostico.conf_int(alpha=0.05)

tabla_pronostico = pd.DataFrame(index=pronostico_log.index)

# Media condicional corregida bajo el supuesto de normalidad predictiva en logaritmos.
# La prueba ARCH detectó heterocedasticidad, por lo que esta corrección es aproximada.
varianza_pronostico = resultado_pronostico.var_pred_mean
tabla_pronostico["Pronóstico medio corregido"] = np.exp(
    pronostico_log + 0.5 * varianza_pronostico
)

# Transformar a la escala original los límites del intervalo predictivo.
tabla_pronostico["Límite inferior 95%"] = np.exp(intervalos_log.iloc[:, 0])
tabla_pronostico["Límite superior 95%"] = np.exp(intervalos_log.iloc[:, 1])
tabla_pronostico.index.name = "Fecha"

print("PRONÓSTICO DEL PRECIO REAL DEL ORO")
print(tabla_pronostico.round(2).to_string())

# Graficar observaciones, pronóstico e intervalos.
plt.figure(figsize=(12, 5))
plt.plot(serie_oro.index, serie_oro, label="Precio observado")
plt.plot(
    tabla_pronostico.index,
    tabla_pronostico["Pronóstico medio corregido"],
    label="Pronóstico medio corregido",
)
plt.fill_between(
    tabla_pronostico.index,
    tabla_pronostico["Límite inferior 95%"],
    tabla_pronostico["Límite superior 95%"],
    alpha=0.2,
    label="Intervalo de predicción del 95%",
)
plt.title("Pronóstico del precio real del oro: 10 trimestres")
plt.xlabel("Fecha")
plt.ylabel("Precio real del oro")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% DECIMOTERCERA PARTE. Coeficientes y ecuación del modelo principal

# 1. Mostrar coeficientes, errores estándar y valores p.
tabla_coeficientes = pd.DataFrame(
    {
        "Coeficiente": modelo_final.params,
        "Error estándar": modelo_final.bse,
        "Valor p": modelo_final.pvalues,
    }
)
print("COEFICIENTES DEL ARIMA(1,1,0) SIN DERIVA")
print(
    tabla_coeficientes.to_string(
        formatters={
            "Coeficiente": lambda x: f"{x:.5f}",
            "Error estándar": lambda x: f"{x:.5f}",
            "Valor p": lambda x: f"{x:.3e}",
        }
    )
)

# 2. Presentar la ecuación estimada.
parametros = dict(zip(modelo_final.param_names, modelo_final.params))
phi = parametros["ar.L1"]
varianza_error = parametros["sigma2"]

print("\nECUACIÓN ESTIMADA")
print(f"Δ ln(Y_t) = {phi:.4f} Δ ln(Y_(t-1)) + ε_t")

# 3. Información general del modelo.
print("\nINFORMACIÓN DEL MODELO")
print("Observaciones:", int(modelo_final.nobs))
print("AIC:", round(modelo_final.aic, 4))
print("BIC:", round(modelo_final.bic, 4))
print("Varianza estimada de las innovaciones:", round(varianza_error, 6))
print("Desviación estándar estimada:", round(np.sqrt(varianza_error), 5))

# %% DECIMOCUARTA PARTE. Prueba de significancia del estimador

# 1. Extraer el coeficiente AR(1) estimado
phi = modelo_final.params["ar.L1"]

# 2. Obtener el error estándar, estadístico z y valor p
error_estandar = modelo_final.bse["ar.L1"]
estadistico_z = modelo_final.tvalues["ar.L1"]
valor_p = modelo_final.pvalues["ar.L1"]

# 3. Definir las hipótesis
# H0: el coeficiente AR(1) es igual a cero
# H1: el coeficiente AR(1) es diferente de cero

nivel_significancia = 0.05

# 4. Mostrar los resultados
print("PRUEBA DE SIGNIFICANCIA DEL COEFICIENTE AR(1)")
print("---------------------------------------------")
print(f"Coeficiente estimado: {phi:.5f}")
print(f"Error estándar: {error_estandar:.5f}")
print(f"Estadístico z: {estadistico_z:.4f}")
print(f"Valor p: {valor_p:.3e}")

# 5. Tomar la decisión estadística
if valor_p < nivel_significancia:
    print("\nDecisión: se rechaza H0.")
    print("El coeficiente AR(1) es estadísticamente significativo.")
else:
    print("\nDecisión: no se rechaza H0.")
    print(
        "No hay evidencia suficiente de que el coeficiente AR(1) sea distinto de cero."
    )
