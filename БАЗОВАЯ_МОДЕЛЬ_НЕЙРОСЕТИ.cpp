/******************************************************************************************
 Пример простой многослойной нейронной сети на C++ с использованием библиотеки TensorFlow:
******************************** © MERKULOV E.V. 2024**************************************
******************************************************************************************/


#include <iostream>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

int main() {
  // Создание сессии TensorFlow
  tensorflow::Session* session;
  tensorflow::Status status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
  if (!status.ok()) {
    std::cerr << "Failed to create TensorFlow session: " << status.ToString() << std::endl;
    return 1;
  }

  // Загрузка модели
  const std::string model_path = "path/to/your/model.pb"; // Укажите путь к вашей модели
  tensorflow::GraphDef graph_def;
  status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), model_path, &graph_def);
  if (!status.ok()) {
    std::cerr << "Failed to load model: " << status.ToString() << std::endl;
    return 1;
  }

  // Загрузка модели в сессию
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cerr << "Failed to load model into session: " << status.ToString() << std::endl;
    return 1;
  }

  // Входные данные для тестирования
  const std::vector<float> input_data = {1.0, 2.0, 3.0}; // Пример входных данных

  // Подготовка входных данных
  tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 3}));
  auto input_tensor_mapped = input_tensor.tensor<float, 2>();
  for (int i = 0; i < input_data.size(); i++) {
    input_tensor_mapped(0, i) = input_data[i];
  }

  // Запуск тестирования модели
  std::vector<tensorflow::Tensor> outputs;
  status = session->Run({{"inputs", input_tensor}}, {"output"}, {}, &outputs);
  if (!status.ok()) {
    std::cerr << "Failed to run model: " << status.ToString() << std::endl;
    return 1;
  }

  // Получение результата тестирования
  tensorflow::Tensor output_tensor = outputs[0];
  auto output_tensor_mapped = output_tensor.tensor<float, 2>();
  for (int i = 0; i < output_tensor_mapped.size(); i++) {
    std::cout << "Output " << i << ": " << output_tensor_mapped(0, i) << std::endl;
  }

  // Освобождение ресурсов
  session->Close();

  return 0;
}

//Прежде чем запустить этот код, убедитесь, что у вас установлена библиотека TensorFlow и все необходимые зависимости.
//Обратите внимание, что вам нужно будет заменить `"path/to/your/model.pb"` на путь к вашей собственной модели нейронной сети.
