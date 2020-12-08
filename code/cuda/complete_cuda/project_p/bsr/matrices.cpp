struct matrix_rows_statistic
{
  unsigned int min_elements_in_rows {};
  unsigned int max_elements_in_rows {};
  unsigned int avg_elements_in_rows {};
  double elements_in_rows_std_deviation {};
};
template <typename data_type>
ell_matrix_class<data_type>::ell_matrix_class (csr_matrix_class<data_type> &matrix)
  : meta (matrix.meta)
{
  if (meta.matrix_storage_scheme != matrix_market::matrix_class::storage_scheme::general)
    throw std::runtime_error ("Only general matrices are supported");

  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ptr = matrix.columns.get ();

  auto [min_elements, max_elements, avg_elements, std_deviation] = get_rows_statistics (meta, row_ptr);
  elements_in_rows = max_elements;

  std::cout << "ELL: " << elements_in_rows
            << " elements in rows (min: " << min_elements
            << "; avg: " << avg_elements<< ")" << std::endl;

  const size_t elements_count = elements_in_rows * meta.rows_count;
  data.reset (new data_type[elements_count]);
  columns.reset (new unsigned int[elements_count]);

  std::fill_n (data.get (), elements_count, 0);
  std::fill_n (columns.get (), elements_count, 0);

  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
    {
      data[row + (element - start) * meta.rows_count] = matrix.data[element];
      columns[row + (element - start) * meta.rows_count] = col_ptr[element];
    }
  }
}

template <typename data_type>
ell_matrix_class<data_type>::ell_matrix_class (csr_matrix_class<data_type> &matrix, unsigned int elements_in_row_arg)
  : meta (matrix.meta)
  , elements_in_rows (elements_in_row_arg)
{
  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ptr = matrix.columns.get ();

  const unsigned int elements_count = get_matrix_size ();
  data.reset (new data_type[elements_count]);
  columns.reset (new unsigned int[elements_count]);

  std::fill_n (data.get (), elements_count, 0);
  std::fill_n (columns.get (), elements_count, 0);

  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    /// Skip extra elements
    for (auto element = start; element < std::min (start + elements_in_row_arg, end); element++)
    {
      data[row + (element - start) * meta.rows_count] = matrix.data[element];
      columns[row + (element - start) * meta.rows_count] = col_ptr[element];
    }
  }
}
template <typename data_type>
coo_matrix_class<data_type>::coo_matrix_class(csr_matrix_class<data_type> &matrix)
  : meta (matrix.meta)
  , elements_count (meta.non_zero_count)
{
  if (meta.matrix_storage_scheme != matrix_market::matrix_class::storage_scheme::general)
    throw std::runtime_error ("Only general matrices are supported");

  data.reset (new data_type[meta.non_zero_count]);
  cols.reset (new unsigned int[meta.non_zero_count]);
  rows.reset (new unsigned int[meta.non_zero_count]);

  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ptr = matrix.columns.get ();

  unsigned int id = 0;
  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
    {
      data[id] = matrix.data[element];
      cols[id] = col_ptr[element];
      rows[id] = row;
      id++;
    }
  }
}

template <typename data_type>
coo_matrix_class<data_type>::coo_matrix_class (csr_matrix_class<data_type> &matrix, unsigned int element_start)
  : meta (matrix.meta)
{
  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ptr = matrix.columns.get ();

  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
      if (element - start >= element_start)
      elements_count++;
  }

  data.reset (new data_type[get_matrix_size ()]);
  cols.reset (new unsigned int[get_matrix_size ()]);
  rows.reset (new unsigned int[get_matrix_size ()]);

  unsigned int id = 0;
  for (unsigned int row = 0; row < meta.rows_count; row++)
  {
    const auto start = row_ptr[row];
    const auto end = row_ptr[row + 1];

    for (auto element = start; element < end; element++)
    {
      if (element - start >= element_start)
      {
        data[id] = matrix.data[element];
        cols[id] = col_ptr[element];
        rows[id] = row;
        id++;
      }
    }
  }
}

template <typename data_type>
class csr_matrix_class
{
public:
  csr_matrix_class () = delete;
  explicit csr_matrix_class (const matrix_market::matrix_class &matrix, bool row_ptr_only=false);


  size_t get_matrix_size () const;

public:
  std::unique_ptr<data_type[]> data;
  std::unique_ptr<unsigned int[]> columns;
  std::unique_ptr<unsigned int[]> row_ptr;
};

template <typename data_type>
class hybrid_matrix_class
{
public:
  hybrid_matrix_class () = delete;
  explicit hybrid_matrix_class (csr_matrix_class<data_type> &matrix);

  void allocate (csr_matrix_class<data_type> &matrix, double percent);


  std::unique_ptr<ell_matrix_class<data_type>> ell_matrix;
  std::unique_ptr<coo_matrix_class<data_type>> coo_matrix;
};
template <typename data_type>
hybrid_matrix_class<data_type>::hybrid_matrix_class (csr_matrix_class<data_type> &matrix)
{

}

template <typename data_type>
void hybrid_matrix_class<data_type>::allocate(csr_matrix_class<data_type> &matrix, double percent)
{
  const auto row_ptr = matrix.row_ptr.get ();
  const auto row_ptr= matrix.

  auto [_1, max_elements, avg_elements, _2] = get_rows_statistics (,row_ptr);
  const unsigned int elements_per_ell = avg_elements + (max_elements - avg_elements) * percent;

  ell_matrix = std::make_unique<ell_matrix_class<data_type>> (matrix, elements_per_ell); /// Don't use more than avg elements in an ELL row
  coo_matrix = std::make_unique<coo_matrix_class<data_type>> (matrix, elements_per_ell); /// Don't use elements before avg elements in an COO row

  std::cout << "ELL elements per row: " << elements_per_ell << "; "
            << "COO elements: " << coo_matrix->get_matrix_size () << "; ELL elements: " << ell_matrix->get_matrix_size () << "; "
            << "COO/ELL Ratio: " << static_cast<double> (coo_matrix->get_matrix_size ()) / ell_matrix->get_matrix_size () << std::endl;
}
template class csr_matrix_class<float>;
template class csr_matrix_class<double>;

template class ell_matrix_class<float>;
template class ell_matrix_class<double>;

template class coo_matrix_class<float>;
template class coo_matrix_class<double>;

template class hybrid_matrix_class<float>;
template class hybrid_matrix_class<double>;
