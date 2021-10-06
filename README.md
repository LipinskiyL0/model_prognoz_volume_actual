# model_prognoz_volume_actual
Модель формирует прогноз объемов продаж

Общая схема работы:
    get_prognoz.get_prognoz(max_date) - принимает max_date - максимальная дата периода обучения 
                                        max_date - параметр, который нужен для корректного рассчета
                                        в тех случаях, когда за последние дни не было продаж и объемы по нулям. 
                                        Что бы эти нули появились в выборке нужно дату определять не по выборке, 
                                        а указать в качестве параметра
                                        get_prognoz.get_prognoz(max_date) - подключаемся к выборке и извлекаем по очереди 
                                        выборку по конкретному product_id и warehouse_id (df4). Формируем выборку под прогноз 
                                        и вызываем my_pipeline_linear.predict_model_linear(df4, period, max_date)
    
    my_pipeline_linear.predict_model_linear(df4, period, max_date) - вычисляет прогнозное значение. df4 выборка по конкретному товару
                                        с конкретного склада. period - период управления=периоду на который делаем заказ, измеряется в днях.
                                        max_date - максимальная дата периода обучения. Для построения прогноза последовательно вызываются:
                                        get_period.get_period(max_date)
                                        agr_period.agr_period(period=period)
                                        get_glubina.get_glubina(n_glub=n_glub)
                                        my_model.my_model(name_model=name_model, n_test=5)
    
    get_period.get_period(max_date) - Агрегация данных о продажах посуточно
    agr_period.agr_period(period=period) - Агрегируем данные по заданному периоду управления
    get_glubina.get_glubina(n_glub=n_glub) - Разрезаем выборку в глубину
    my_model.my_model(name_model=name_model, n_test=5) - строим модель и вычисляем выход
    
 