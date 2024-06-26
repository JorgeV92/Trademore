module Alpaca = struct

    open Lwt
    open Cohttp
    open Cohttp_lwt_unix
    open Yojson.Basic.Util

    (* Alpaca API Configuration *)
    let alpaca_endpoint = "https://paper-api.alpaca.markets"
    let alpaca_market_endpoint = "https://data.alpaca.markets"
    let alpaca_key = "PKQ6ZYP9Q3HLDCVJB1TV"
    let alpaca_secret = "wjobG7qsUWMRfXlg6uGpWjmKWdqUf2MdTavc4OWA"

    (* Initialize Headers *)
    let init_headers () =
      Header.init ()
      |> fun h -> Header.add_list h [("APCA-API-KEY-ID", alpaca_key);
                                    ("APCA-API-SECRET-KEY", alpaca_secret);
                                    ("Content-Type", "application/json")]

    (* Get Account Info *)
    let get_account_info () =
      let uri = Uri.of_string (alpaca_endpoint ^ "/v2/account") in
      Client.call ~headers:(init_headers ()) `GET uri
      >>= fun (res, body) ->
      let code = res |> Response.status |> Code.code_of_status in
      body |> Cohttp_lwt.Body.to_string >|= fun body ->
      (code, body)

    (* Process Account Info JSON *)
    let process_account_info_json json =
      let status = json |> member "status" |> to_string in
      let buying_power = json |> member "buying_power" |> to_string in
      let portfolio_value = json |> member "portfolio_value" |> to_string in
      let cash = json |> member "cash" |> to_string in
      let equity = json |> member "equity" |> to_string in
      let daytrading_buying_power = json |> member "daytrading_buying_power" |> to_string in
      (status, buying_power, portfolio_value, cash, equity, daytrading_buying_power)

    (* Run Account Info *)
    let run_account_info () =
      get_account_info ()
      >>= fun (code, body) ->
      if code = 200 then
        let json = Yojson.Basic.from_string body in
        let (status, buying_power, portfolio_value, cash, equity, daytrading_buying_power) =
          process_account_info_json json in
        Lwt_io.printf "\nAccount Status: %s\nBuying Power: %s\nPortfolio Value: %s\nCash: %s\nEquity: %s\nDay Trading Buying Power: %s\n"
          status buying_power portfolio_value cash equity daytrading_buying_power
      else
        Lwt_io.printl "Error fetching account info"

    (* Get Current Date *)
    let current_date () =
      let open Ptime in
      let date, _ = Ptime_clock.now () |> to_date_time in
      let year, month, day = date in
      Printf.sprintf "%04d-%02d-%02d" year month day

    (* Get Bars for a Symbol *)
    let get_bars symbol =
      let uri = Uri.of_string (alpaca_market_endpoint ^ "/v2/stocks/" ^ symbol ^ "/bars") in
      let query = [("start", "2021-05-15T09:30:00-04:00");
                  ("end", current_date () ^ "T16:00:00-04:00");
                  ("timeframe", "1Day");
                  ("limit", "5000")] in
      let uri = Uri.with_query' uri query in
      Client.call ~headers:(init_headers ()) `GET uri
      >>= fun (res, body) ->
      let code = res |> Response.status |> Code.code_of_status in
      body |> Cohttp_lwt.Body.to_string >|= fun body ->
      (code, body, symbol)

    (* Get Bars for Multiple Symbols *)
    let get_bars_multiple symbols =
      Lwt_list.map_s get_bars symbols

    (* Run Get Bars for Symbols *)
    let run_get_bars_for_symbols symbols =
      let responses = Lwt_main.run (get_bars_multiple symbols) in
      Lwt_list.iter_s (fun (code, body, symbol) ->
        if code = 200 then
          Lwt_io.printf "Success for %s: %s\n" symbol body
        else
          Lwt_io.printf "Error for %s: code %d, body: %s\n" symbol code body
      ) responses

    (* Place an Order *)
    let place_order symbol qty side type_ time_in_force =
      let uri = Uri.of_string (alpaca_endpoint ^ "/v2/orders") in
      let body = `Assoc [("symbol", `String symbol);
                        ("qty", `Int qty);
                        ("side", `String side);
                        ("type", `String type_);
                        ("time_in_force", `String time_in_force)]
                |> Yojson.Basic.to_string in
      Client.call ~headers:(init_headers ()) ~body:(`String body) `POST uri
      >>= fun (res, body) ->
      let code = res |> Response.status |> Code.code_of_status in
      body |> Cohttp_lwt.Body.to_string >|= fun body ->
      (code, body)
      
end